"""外部API・GitHubの変更なしで、実LanceDBの配布・復元とRelease操作を検証する。"""

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import lancedb
from lancedb.index import FTS

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "scripts/rag_indexes"))
sys.path.insert(0, str(PROJECT))

import cli as distribution  # noqa: E402
import validate as index_validation  # noqa: E402
from app.advanced_rag import fulltext  # noqa: E402

SHA = "a" * 40


class DistributionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.payload = self.work / "payload"
        self.payload.mkdir()
        db = lancedb.connect(self.payload)
        for index in distribution.INDEXES:
            records = [
                {
                    "id": f"{index}-1",
                    "title": "顧客検索",
                    "text": "顧客情報の検索手順",
                    "collection": "資料A",
                },
                {
                    "id": f"{index}-2",
                    "title": "登録",
                    "text": "新しい顧客の登録方法",
                    "collection": "資料B",
                },
            ]
            rows = [
                {**record, **fulltext.columns(record["text"]), "vector": vector}
                for record, vector in zip(records, [[1.0, 0.0], [0.0, 1.0]])
            ]
            table = db.create_table(f"docs_{index}", rows)
            for column in fulltext.ALL_COLUMNS:
                table.create_index(
                    column, config=FTS(base_tokenizer="whitespace", with_position=True)
                )
            (self.payload / f"docs_{index}.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            (self.payload / f"docs_{index}.meta.json").write_text(
                json.dumps(
                    {
                        "extractor": index,
                        "embedding_model": "text-embedding-3-small",
                    }
                )
            )
        self.manifest = {
            "format_version": 1,
            "source_sha": SHA,
            "python": "3.13.15",
            "uv": "uv test",
            "packages": {name: version(name) for name in distribution.PACKAGES},
            "indexes": {
                index: {"embedding_model": "text-embedding-3-small", "rows": 2}
                for index in distribution.INDEXES
            },
        }
        (self.payload / "manifest.json").write_text(json.dumps(self.manifest))
        (self.payload / "attributions").mkdir()
        (self.payload / "attributions/README.md").write_text("出典・ライセンス")
        self.archive = self.work / distribution.ASSET
        self.pack()
        self.destination = self.work / "project/data/lancedb"
        self.destination.mkdir(parents=True)
        simple = lancedb.connect(self.destination).create_table(
            "simple_rag", [{"text": "残すデータ"}]
        )
        self.assertEqual(simple.count_rows(), 1)
        shutil.copy2(ROOT / "day2/uv.lock", self.destination.parent.parent / "uv.lock")
        (self.destination.parent.parent / ".python-version").write_text("3.13\n")

    def pack(self, extra=None):
        with tarfile.open(self.archive, "w:gz") as archive:
            for path in self.payload.iterdir():
                archive.add(path, arcname=path.name)
            if extra:
                archive.addfile(*extra)

    def download(self, archive=None):
        args = argparse.Namespace(
            archive=archive or self.archive,
            destination=self.destination,
            version=None,
            repo=distribution.REPO,
        )
        with patch.object(distribution, "output", return_value="uv test"):
            distribution.download(args)

    def assert_simple_preserved(self):
        table = lancedb.connect(self.destination).open_table("simple_rag")
        self.assertEqual(table.search().to_list(), [{"text": "残すデータ"}])

    def test_roundtrip_search_and_preserve_simple_rag(self):
        self.download()
        # 更新時も対象テーブルだけを入れ替え、余分な旧ファイルを残さない。
        (self.destination / "docs_basic.lance/stale").write_text("old")
        self.download()
        self.assertFalse((self.destination / "docs_basic.lance/stale").exists())
        self.assert_simple_preserved()
        indexes = index_validation.validate(self.destination)
        self.assertEqual(
            {name: entry["rows"] for name, entry in indexes.items()},
            {name: 2 for name in distribution.INDEXES},
        )
        self.assertTrue(
            (self.destination / "rag_indexes_attributions/README.md").is_file()
        )
        self.assertEqual(
            json.loads((self.destination / "rag_indexes_manifest.json").read_text())[
                "source_sha"
            ],
            SHA,
        )

    def test_standalone_starter_download_from_another_directory(self):
        starter = self.destination.parent.parent
        scripts = starter / "scripts/rag_indexes"
        scripts.mkdir(parents=True)
        for name in ("download.sh", "cli.py"):
            shutil.copy2(PROJECT / "scripts/rag_indexes" / name, scripts / name)
        result = subprocess.run(
            ["bash", str(scripts / "download.sh"), "--archive", str(self.archive)],
            cwd=self.work,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn(str(self.destination), result.stdout)
        self.assertEqual(
            index_validation.validate(self.destination)["basic"]["rows"], 2
        )
        self.assert_simple_preserved()

    def test_truncated_archive_leaves_existing_data(self):
        self.download()
        original = (self.destination / "docs_basic.meta.json").read_bytes()
        truncated = self.work / "truncated.tar.gz"
        truncated.write_bytes(self.archive.read_bytes()[:-8])
        with self.assertRaises((EOFError, tarfile.TarError, OSError)):
            self.download(truncated)
        self.assertEqual(
            (self.destination / "docs_basic.meta.json").read_bytes(), original
        )
        self.assert_simple_preserved()

    def test_missing_component_leaves_existing_data(self):
        (self.payload / "docs_vision.meta.json").unlink()
        self.pack()
        with self.assertRaises(OSError):
            self.download()
        self.assertFalse((self.destination / "docs_basic.lance").exists())
        self.assert_simple_preserved()

    def test_rejects_traversal_links_and_other_tables(self):
        for name, kind in (
            ("../outside", tarfile.REGTYPE),
            ("docs_basic.lance/link", tarfile.SYMTYPE),
            ("simple_rag.lance/overwrite", tarfile.REGTYPE),
        ):
            with self.subTest(name=name):
                member = tarfile.TarInfo(name)
                member.type = kind
                member.linkname = "../../outside" if kind == tarfile.SYMTYPE else ""
                self.pack((member, io.BytesIO(b"")))
                with self.assertRaises(ValueError):
                    self.download()
                self.assert_simple_preserved()

    def test_incompatible_dependencies_leave_existing_data(self):
        self.manifest["packages"]["lancedb"] = "0.0.0"
        (self.payload / "manifest.json").write_text(json.dumps(self.manifest))
        self.pack()
        with self.assertRaisesRegex(ValueError, "lancedb"):
            self.download()
        self.assertFalse((self.destination / "docs_basic.lance").exists())
        self.assert_simple_preserved()

    def test_rolls_back_when_replacing_a_component_fails(self):
        self.download()
        (self.destination / "docs_basic.lance/old_marker").write_text("old")
        original_rename = Path.rename

        def rename(path, target):
            if path.parent.name == "incoming" and path.name == "docs_structured.jsonl":
                raise OSError("test installation failure")
            return original_rename(path, target)

        with patch.object(Path, "rename", rename), self.assertRaises(OSError):
            distribution.install(self.payload, self.destination)
        self.assertEqual(
            (self.destination / "docs_basic.lance/old_marker").read_text(), "old"
        )
        self.assertEqual(
            index_validation.validate(self.destination)["vision"]["rows"], 2
        )
        self.assert_simple_preserved()

    def test_public_download_routes(self):
        for date, path in (
            (None, "latest/download"),
            ("2026-10-08", "download/2026-10-08"),
        ):
            with self.subTest(version=date):

                def curl(*command):
                    self.assertEqual(
                        command[-1],
                        f"https://github.com/{distribution.REPO}/releases/{path}/{distribution.ASSET}",
                    )
                    shutil.copy2(self.archive, command[command.index("--output") + 1])

                args = argparse.Namespace(
                    archive=None,
                    destination=self.destination,
                    version=date,
                    repo=distribution.REPO,
                )
                with (
                    patch.object(distribution, "run", side_effect=curl),
                    patch.object(distribution, "output", return_value="uv test"),
                ):
                    distribution.download(args)
                self.assert_simple_preserved()

    def test_failed_download_does_not_install(self):
        args = argparse.Namespace(
            archive=None,
            destination=self.destination,
            version=None,
            repo=distribution.REPO,
        )
        with patch.object(
            distribution, "run", side_effect=subprocess.CalledProcessError(22, "curl")
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                distribution.download(args)
        self.assertFalse((self.destination / "docs_basic.lance").exists())
        self.assert_simple_preserved()

    def release_args(self, draft=False):
        return argparse.Namespace(
            archive=self.archive,
            version="2026-10-08",
            repo=distribution.REPO,
            draft=draft,
        )

    def test_create_packages_only_targets_and_checks_restored_search(self):
        root = self.work / "build-repo"
        project = root / "day2"
        db_dir = project / "data/lancedb"
        shutil.copytree(self.payload, db_dir)
        lancedb.connect(db_dir).create_table("simple_rag", [{"text": "梱包しない"}])
        attribution = project / "data/corpus/sample/README.md"
        attribution.parent.mkdir(parents=True)
        attribution.write_text("出典・ライセンス")
        (project / ".env").write_text("VISION_MODEL=test-model\n")
        archive = root / "dist" / distribution.ASSET
        args = argparse.Namespace(
            archive=archive, embedding_model="text-embedding-3-small"
        )

        def git_output(*command):
            return SHA if command[1] == "rev-parse" else ""

        def build_or_validate(*command, **kwargs):
            if str(distribution.VALIDATOR) in command:
                db = (
                    Path(command[command.index("--db-dir") + 1])
                    if "--db-dir" in command
                    else db_dir
                )
                index_validation.validate(db)
                if "--output" in command:
                    Path(command[command.index("--output") + 1]).write_text(
                        json.dumps(self.manifest)
                    )

        with (
            patch.object(distribution, "ROOT", root),
            patch.object(distribution, "PROJECT", project),
            patch.object(distribution, "output", side_effect=git_output),
        ):
            with patch.object(
                distribution, "run", side_effect=build_or_validate
            ) as run:
                distribution.create(args)
        builds = [
            call
            for call in run.call_args_list
            if "app.advanced_rag.build_index" in call.args
        ]
        self.assertEqual(
            [call.args[call.args.index("--extractor") + 1] for call in builds],
            list(distribution.INDEXES),
        )
        self.assertTrue(any("--db-dir" in call.args for call in run.call_args_list))
        self.assertTrue(all("--env-file" in call.args for call in builds))
        restored = self.work / "create-restored"
        restored.mkdir()
        distribution.unpack(archive, restored)
        self.assertFalse((restored / "simple_rag.lance").exists())

    def test_manifest_generation_requires_no_api_keys(self):
        manifest = self.work / "generated-manifest.json"
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"OPENAI_API_KEY", "WANDB_PROJECT", "WANDB_API_KEY"}
        }
        environment["VISION_MODEL"] = "test-vision-model"
        environment["VISION_REASONING_EFFORT"] = "low"
        result = subprocess.run(
            [
                sys.executable,
                str(distribution.VALIDATOR),
                "--db-dir",
                str(self.payload),
                "--source-sha",
                SHA,
                "--output",
                str(manifest),
            ],
            cwd=self.work,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("検証OK", result.stdout)
        recorded = json.loads(manifest.read_text())
        self.assertEqual(recorded["source_sha"], SHA)
        self.assertEqual(recorded["vision_model"], "test-vision-model")
        self.assertEqual(recorded["indexes"]["vision"]["vector_dimensions"], 2)

    def test_create_refuses_dirty_source_before_build(self):
        args = argparse.Namespace(
            archive=self.archive, embedding_model="text-embedding-3-small"
        )
        with patch.object(distribution, "output", side_effect=[SHA, " M source.py"]):
            with patch.object(distribution, "run") as run:
                with self.assertRaises(ValueError):
                    distribution.create(args)
                run.assert_not_called()

    def test_release_uses_build_sha_and_publishes_after_upload(self):
        with (
            patch.object(distribution, "api", side_effect=[{"sha": SHA}, None, None]),
            patch.object(distribution, "run") as run,
        ):
            distribution.release_create(self.release_args())
        commands = [call.args for call in run.call_args_list]
        self.assertEqual(commands[0][2], "create")
        self.assertEqual(commands[0][commands[0].index("--target") + 1], SHA)
        self.assertIn("--draft", commands[0])
        self.assertEqual(commands[1][2], "upload")
        self.assertEqual(Path(commands[1][4]).name, distribution.ASSET)
        self.assertEqual(commands[2][2], "edit")
        self.assertIn("--draft=false", commands[2])

    def test_draft_is_reused_without_another_release(self):
        existing = {"draft": True, "target_commitish": SHA}
        with (
            patch.object(
                distribution, "api", side_effect=[{"sha": SHA}, None, existing]
            ),
            patch.object(distribution, "run") as run,
        ):
            distribution.release_create(self.release_args(draft=True))
        self.assertEqual(len(run.call_args_list), 1)
        self.assertEqual(run.call_args.args[2], "upload")

    def test_upload_failure_does_not_publish(self):
        def gh(*command):
            if command[2] == "upload":
                raise subprocess.CalledProcessError(1, "gh release upload")

        with (
            patch.object(distribution, "api", side_effect=[{"sha": SHA}, None, None]),
            patch.object(distribution, "run", side_effect=gh) as run,
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                distribution.release_create(self.release_args())
        self.assertNotIn("edit", [call.args[2] for call in run.call_args_list])

    def test_rejects_tag_mismatch_and_published_release(self):
        cases = (
            [{"sha": SHA}, {"object": {}}, {"sha": "b" * 40}],
            [{"sha": SHA}, None, {"draft": False}],
            [{"sha": SHA}, None, {"draft": True, "target_commitish": "b" * 40}],
        )
        for replies in cases:
            with self.subTest(replies=replies):
                with (
                    patch.object(distribution, "api", side_effect=replies),
                    patch.object(distribution, "run") as run,
                ):
                    with self.assertRaises(ValueError):
                        distribution.release_create(self.release_args())
                    run.assert_not_called()

    def test_delete_includes_tag_and_only_skips_prompt_with_yes(self):
        for yes in (False, True):
            with (
                patch.object(distribution, "run") as run,
                patch.object(distribution, "api", return_value={"object": {}}),
            ):
                distribution.release_delete(
                    argparse.Namespace(
                        version="2026-10-08", repo=distribution.REPO, yes=yes
                    )
                )
            self.assertIn("--cleanup-tag", run.call_args.args)
            self.assertEqual("--yes" in run.call_args.args, yes)

    def test_delete_draft_without_a_tag(self):
        with (
            patch.object(distribution, "run") as run,
            patch.object(distribution, "api", return_value=None),
        ):
            distribution.release_delete(
                argparse.Namespace(
                    version="2026-10-08", repo=distribution.REPO, yes=True
                )
            )
        self.assertEqual(run.call_args.args[2], "delete")
        self.assertNotIn("--cleanup-tag", run.call_args.args)


class ArgumentsTest(unittest.TestCase):
    def test_required_versions_and_invalid_dates(self):
        for args in (
            ("release-create",),
            ("release-delete",),
            ("release-delete", "--version", "2026-02-30"),
            ("download", "--version", "latest"),
            ("download", "--version", "2026-10-08", "--archive", "x"),
        ):
            result = subprocess.run(
                [sys.executable, str(PROJECT / "scripts/rag_indexes/cli.py"), *args],
                capture_output=True,
            )
            self.assertEqual(result.returncode, 2, args)

    def test_api_does_not_hide_authentication_or_server_errors(self):
        for code in (401, 403, 500):
            result = subprocess.CompletedProcess([], 1, "", f"gh: failed (HTTP {code})")
            with patch.object(distribution.subprocess, "run", return_value=result):
                with self.assertRaises(RuntimeError):
                    distribution.api(distribution.REPO, "x", optional=True)
        missing = subprocess.CompletedProcess([], 1, "", "gh: Not Found (HTTP 404)")
        with patch.object(distribution.subprocess, "run", return_value=missing):
            self.assertIsNone(distribution.api(distribution.REPO, "x", optional=True))


if __name__ == "__main__":
    unittest.main()
