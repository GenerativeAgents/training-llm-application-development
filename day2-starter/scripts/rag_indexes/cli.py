"""インデックスの生成・梱包・取得・復元。取得と復元には標準ライブラリだけを使う。"""

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import tomllib

REPO = "GenerativeAgents/training-llm-application-development"
PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
VALIDATOR = Path(__file__).resolve().with_name("validate.py")
ASSET = "advanced-rag-indexes.tar.gz"
INDEXES = ("basic", "structured", "vision")
COMPONENTS = tuple(
    f"docs_{index}{suffix}"
    for index in INDEXES
    for suffix in (".lance", ".jsonl", ".meta.json")
)
PACKAGES = ("lancedb", "pyarrow", "sudachipy", "sudachidict-core")


def run(*command: str, cwd: Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def output(*command: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def version(value: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise argparse.ArgumentTypeError("--version は yyyy-mm-dd で指定してください")
    try:
        dt.date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("実在する日付を指定してください") from error
    return value


def repository(value: str) -> str:
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", value):
        raise argparse.ArgumentTypeError("--repo は owner/repo で指定してください")
    return value


def read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("format_version") != 1 or not re.fullmatch(
        r"[0-9a-f]{40}", manifest.get("source_sha", "")
    ):
        raise ValueError("配布物の形式または生成元コミットが不正です")
    if set(manifest.get("indexes", {})) != set(INDEXES):
        raise ValueError("配布物には basic / structured / vision が必要です")
    if not all(manifest.get("packages", {}).get(name) for name in PACKAGES):
        raise ValueError("配布物に依存バージョンの記録がありません")
    return manifest


def unpack(archive: Path, destination: Path) -> dict[str, Any]:
    """許可した通常ファイル・ディレクトリだけを一時領域に展開する。"""
    allowed = {*COMPONENTS, "manifest.json", "attributions", "README.md"}
    with tarfile.open(archive, "r:gz") as bundle:
        seen = set()
        members = bundle.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"配布物のパスが不正です: {member.name}")
            name = path.as_posix()
            top = path.parts[0]
            if (
                name in seen
                or top not in allowed
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"配布物に未対応の項目があります: {member.name}")
            if len(path.parts) > 1 and not (
                top.endswith(".lance") or top == "attributions"
            ):
                raise ValueError(f"配布物の階層が不正です: {member.name}")
            seen.add(name)
        # gzip末尾まで読み、途中で途切れた取得も配置前に検出する。
        while bundle.fileobj.read(1024 * 1024):
            pass
        bundle.extractall(destination, members=members, filter="data")
    manifest = read_manifest(destination / "manifest.json")
    for index in INDEXES:
        name = f"docs_{index}"
        table = destination / f"{name}.lance"
        if (
            not table.is_dir()
            or not (table / "_versions").is_dir()
            or not (table / "data").is_dir()
        ):
            raise ValueError(f"{name} のLanceテーブルがありません")
        if not (table / "_indices").is_dir():
            raise ValueError(f"{name} の全文検索インデックスがありません")
        meta = json.loads((destination / f"{name}.meta.json").read_text())
        recorded = manifest["indexes"][index]
        if (
            meta.get("extractor") != index
            or meta.get("embedding_model") != recorded["embedding_model"]
        ):
            raise ValueError(f"{name} の生成条件が一致しません")
        count = 0
        with (destination / f"{name}.jsonl").open() as records:
            for line in records:
                json.loads(line)
                count += 1
        if not count or count != recorded["rows"]:
            raise ValueError(f"{name} の文書数が一致しません")
    if not (destination / "attributions").is_dir() or not list(
        (destination / "attributions").rglob("README.md")
    ):
        raise ValueError("配布物に出典・ライセンスのREADMEがありません")
    return manifest


def check_environment(manifest: dict[str, Any], project: Path) -> None:
    with (project / "uv.lock").open("rb") as lock_file:
        lock = tomllib.load(lock_file)
    locked = {p["name"]: p["version"] for p in lock["package"]}
    for name in PACKAGES:
        if locked.get(name) != manifest["packages"][name]:
            raise ValueError(
                f"{name} の版が配布物と異なります。同じReleaseの教材ソースを使ってください"
            )
    python_version = (project / ".python-version").read_text().strip()
    if not manifest["python"].startswith(f"{python_version}."):
        raise ValueError(
            "Pythonの版が配布物と異なります。同じReleaseの教材ソースを使ってください"
        )
    if output("uv", "--version") != manifest["uv"]:
        print(
            f"参考: 生成時のuvは {manifest['uv']} です。現在は {output('uv', '--version')} です。"
        )


def install(source: Path, destination: Path) -> None:
    """管理対象だけを交換し、交換中に失敗した場合は元へ戻す。"""
    destination.mkdir(parents=True, exist_ok=True)
    entries = {name: name for name in COMPONENTS}
    entries.update(
        {
            "manifest.json": "rag_indexes_manifest.json",
            "attributions": "rag_indexes_attributions",
        }
    )
    with tempfile.TemporaryDirectory(
        prefix=".rag-indexes-", dir=destination
    ) as staging_dir:
        staging = Path(staging_dir)
        incoming = staging / "incoming"
        backup = staging / "backup"
        incoming.mkdir()
        backup.mkdir()
        for original, target in entries.items():
            path = source / original
            if path.is_dir():
                shutil.copytree(path, incoming / target)
            else:
                shutil.copy2(path, incoming / target)
        replaced = []
        saved = []
        try:
            for target in entries.values():
                current = destination / target
                if current.exists() or current.is_symlink():
                    current.rename(backup / target)
                    saved.append(target)
                (incoming / target).rename(current)
                replaced.append(target)
        except BaseException:
            for target in reversed(replaced):
                (destination / target).rename(incoming / target)
            for target in reversed(saved):
                (backup / target).rename(destination / target)
            raise


def download(args: argparse.Namespace) -> None:
    destination = args.destination.resolve()
    project = destination.parent.parent
    with tempfile.TemporaryDirectory(prefix="rag-indexes-") as temporary:
        work = Path(temporary)
        archive = args.archive
        if archive is None:
            archive = work / ASSET
            route = f"download/{args.version}" if args.version else "latest/download"
            url = f"https://github.com/{args.repo}/releases/{route}/{ASSET}"
            print(f"取得: {url}", flush=True)
            run(
                "curl",
                "--fail",
                "--location",
                "--retry",
                "3",
                "--output",
                str(archive),
                url,
            )
        payload = work / "payload"
        payload.mkdir()
        manifest = unpack(archive, payload)
        check_environment(manifest, project)
        install(payload, destination)
    print(f"配置完了: {destination} (生成元 {manifest['source_sha']})")


def create(args: argparse.Namespace) -> None:
    sha = output("git", "rev-parse", "HEAD")
    if output("git", "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError(
            "生成元を記録するため、変更と追加ファイルをコミットしてから生成してください"
        )
    project = PROJECT
    run("uv", "sync", "--frozen", "--project", str(project))
    uv_run = ["uv", "run", "--frozen"]
    # 抽出器のimportより前にAPIキー・Vision設定を読み込む必要がある。
    if (project / ".env").is_file():
        uv_run.extend(["--env-file", str(project / ".env")])
    for index in INDEXES:
        run(
            *uv_run,
            "python",
            "-m",
            "app.advanced_rag.build_index",
            "--extractor",
            index,
            "--embedding-model",
            args.embedding_model,
            cwd=project,
        )
    if output("git", "rev-parse", "HEAD") != sha or output(
        "git", "status", "--porcelain", "--untracked-files=normal"
    ):
        raise ValueError("生成中にソースが変わりました。再生成してください")
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="rag-indexes-", dir=args.archive.parent
    ) as temporary:
        work = Path(temporary)
        run(
            *uv_run,
            "python",
            str(VALIDATOR),
            "--source-sha",
            sha,
            "--output",
            str(work / "manifest.json"),
            cwd=project,
        )
        attributions = work / "attributions"
        for attribution in (project / "data/corpus").rglob("README.md"):
            target = attributions / attribution.relative_to(project / "data/corpus")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(attribution, target)
        (work / "README.md").write_text(
            "# Advanced RAGインデックス\n\n"
            "原資料から抽出・分割したテキスト、図のAI説明、埋め込み、全文検索用のデータです。\n"
            "出典・著作権・ライセンスは attributions/ 内の各READMEを参照してください。\n"
            "生成条件は manifest.json を参照してください。\n"
        )
        archive = work / ASSET
        with tarfile.open(archive, "w:gz") as bundle:
            for name in COMPONENTS:
                bundle.add(project / "data/lancedb" / name, arcname=name)
            for name in ("manifest.json", "attributions", "README.md"):
                bundle.add(work / name, arcname=name)
        # 配布物から復元して検索も検証する。生成済みDBを読めるだけでは不十分。
        restored = work / "restored"
        restored.mkdir()
        unpack(archive, restored)
        run(
            "uv",
            "run",
            "--frozen",
            "python",
            str(VALIDATOR),
            "--db-dir",
            str(restored),
            cwd=project,
        )
        os.replace(archive, args.archive)
    print(f"配布物: {args.archive} (生成元 {sha})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "create",
        prog="scripts/rag_indexes/create.sh",
        description="3種類を再生成・検証・梱包 (OpenAI API利用あり)",
    )
    build.add_argument("--embedding-model", default="text-embedding-3-small")
    build.add_argument("--archive", type=Path, default=ROOT / "dist" / ASSET)
    build.set_defaults(action=create)
    get = commands.add_parser(
        "download",
        prog="scripts/rag_indexes/download.sh",
        description="指定日付のRelease、省略時はLatestを配置",
    )
    inputs = get.add_mutually_exclusive_group()
    inputs.add_argument("--version", type=version)
    inputs.add_argument("--archive", type=Path, help="ローカルの配布物で復元を試す")
    get.add_argument("--destination", type=Path, default=PROJECT / "data/lancedb")
    get.add_argument("--repo", type=repository, default=REPO)
    get.set_defaults(action=download)
    args = parser.parse_args()
    if getattr(args, "archive", None) is not None:
        args.archive = args.archive.resolve()
    try:
        args.action(args)
    except (
        OSError,
        EOFError,
        ValueError,
        KeyError,
        RuntimeError,
        tarfile.TarError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
