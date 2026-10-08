"""教材の日付Releaseを作成・削除する。GitHub操作には標準ライブラリとghを使う。"""

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from day2.scripts.rag_indexes import cli as indexes  # noqa: E402


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def api(repo: str, path: str, *, optional: bool = False) -> dict[str, Any] | None:
    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/{path}"], text=True, capture_output=True, cwd=ROOT
    )
    if result.returncode:
        if optional and "(HTTP 404)" in result.stderr:
            return None
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def create(args: argparse.Namespace) -> None:
    # 添付するファイルの生成元を使う。実行時のHEADをタグの作成元にしない。
    with tempfile.TemporaryDirectory(prefix="rag-indexes-") as temporary:
        manifest = indexes.unpack(args.archive, Path(temporary))
    sha = manifest["source_sha"]
    commit = api(args.repo, f"commits/{sha}")
    if commit is None or commit["sha"] != sha:
        raise ValueError(
            "生成元コミットをGitHubにpushしてからReleaseを作成してください"
        )
    tag = api(args.repo, f"git/ref/tags/{args.version}", optional=True)
    if tag is not None:
        tagged_commit = api(args.repo, f"commits/{args.version}")
        if tagged_commit is None or tagged_commit["sha"] != sha:
            raise ValueError(
                "同名タグが別のコミットを指しています。別の日付を使うか検証用Release・タグを削除してください"
            )
    release = api(args.repo, f"releases/tags/{args.version}", optional=True)
    if release is not None and not release["draft"]:
        raise ValueError("公開済みの同名Releaseは更新しません")
    if release is not None:
        # Draftの作成時にはタグがまだ無い場合もあるので、記録した生成元も照合する。
        if release["target_commitish"] != sha and tag is None:
            raise ValueError("既存Draftの生成元コミットが異なります")
    else:
        run(
            "gh",
            "release",
            "create",
            args.version,
            "--repo",
            args.repo,
            "--target",
            sha,
            "--title",
            args.version,
            "--notes",
            f"教材ソースとAdvanced RAGインデックス。生成元: {sha}",
            "--draft",
        )
    # 添付失敗時にはDraftのまま残す。別名の入力でもReleaseでは固定のasset名にする。
    with tempfile.TemporaryDirectory(prefix="rag-indexes-") as temporary:
        asset = Path(temporary) / indexes.ASSET
        shutil.copy2(args.archive, asset)
        run(
            "gh",
            "release",
            "upload",
            args.version,
            str(asset),
            "--repo",
            args.repo,
            "--clobber",
        )
    if not args.draft:
        run(
            "gh",
            "release",
            "edit",
            args.version,
            "--repo",
            args.repo,
            "--draft=false",
            "--latest",
        )
    print(
        f"{'Draft' if args.draft else '公開'}: https://github.com/{args.repo}/releases/tag/{args.version}"
    )


def delete(args: argparse.Namespace) -> None:
    tag = api(args.repo, f"git/ref/tags/{args.version}", optional=True)
    command = [
        "gh",
        "release",
        "delete",
        args.version,
        "--repo",
        args.repo,
    ]
    # DraftにはまだGitタグが無い場合がある。無いタグの削除で失敗させない。
    if tag is not None:
        command.append("--cleanup-tag")
    if args.yes:
        command.append("--yes")
    run(*command)
    print(
        "Release・添付・リモートタグを削除しました。ローカルタグとソースコミットは残ります。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser(
        "create",
        prog="scripts/release/create.sh",
        description="日付Releaseに配布物を添付。既存Draftは公開可能",
    )
    publish.add_argument("--version", type=indexes.version, required=True)
    publish.add_argument("--draft", action="store_true")
    publish.add_argument("--archive", type=Path, default=ROOT / "dist" / indexes.ASSET)
    publish.add_argument("--repo", type=indexes.repository, default=indexes.REPO)
    publish.set_defaults(action=create)
    remove = commands.add_parser(
        "delete",
        prog="scripts/release/delete.sh",
        description="指定日付のRelease・添付・リモートタグを削除",
    )
    remove.add_argument("--version", type=indexes.version, required=True)
    remove.add_argument("--yes", action="store_true", help="ghの削除確認を省略")
    remove.add_argument("--repo", type=indexes.repository, default=indexes.REPO)
    remove.set_defaults(action=delete)
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
