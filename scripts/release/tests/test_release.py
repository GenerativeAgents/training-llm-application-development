"""実際のGitHub操作なしで、教材Releaseの作成・削除を検証する。"""

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.release import cli as release  # noqa: E402

SHA = "a" * 40


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        work = Path(self.temporary.name)
        payload = work / "payload"
        payload.mkdir()
        # Release側が確認するアーカイブ構成を用意する。検索検証はday2側のテストで行う。
        for index in release.indexes.INDEXES:
            for name in ("_versions", "data", "_indices"):
                (payload / f"docs_{index}.lance" / name).mkdir(parents=True)
            (payload / f"docs_{index}.jsonl").write_text('{"text": "sample"}\n')
            (payload / f"docs_{index}.meta.json").write_text(
                json.dumps({"extractor": index, "embedding_model": "test-model"})
            )
        (payload / "manifest.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "source_sha": SHA,
                    "packages": {
                        name: "test-version" for name in release.indexes.PACKAGES
                    },
                    "indexes": {
                        index: {"embedding_model": "test-model", "rows": 1}
                        for index in release.indexes.INDEXES
                    },
                }
            )
        )
        (payload / "attributions").mkdir()
        (payload / "attributions/README.md").write_text("出典・ライセンス")
        self.archive = work / release.indexes.ASSET
        with tarfile.open(self.archive, "w:gz") as bundle:
            for path in payload.iterdir():
                bundle.add(path, arcname=path.name)

    def release_args(self, draft=False):
        return argparse.Namespace(
            archive=self.archive,
            version="2026-10-08",
            repo=release.indexes.REPO,
            draft=draft,
        )

    def test_release_uses_build_sha_and_publishes_after_upload(self):
        with (
            patch.object(release, "api", side_effect=[{"sha": SHA}, None, None]),
            patch.object(release, "run") as run,
        ):
            release.create(self.release_args())
        commands = [call.args for call in run.call_args_list]
        self.assertEqual(commands[0][2], "create")
        self.assertEqual(commands[0][commands[0].index("--target") + 1], SHA)
        self.assertIn("--draft", commands[0])
        self.assertEqual(commands[1][2], "upload")
        self.assertEqual(Path(commands[1][4]).name, release.indexes.ASSET)
        self.assertEqual(commands[2][2], "edit")
        self.assertIn("--draft=false", commands[2])

    def test_draft_is_reused_without_another_release(self):
        existing = {"draft": True, "target_commitish": SHA}
        with (
            patch.object(release, "api", side_effect=[{"sha": SHA}, None, existing]),
            patch.object(release, "run") as run,
        ):
            release.create(self.release_args(draft=True))
        self.assertEqual(len(run.call_args_list), 1)
        self.assertEqual(run.call_args.args[2], "upload")

    def test_upload_failure_does_not_publish(self):
        def gh(*command):
            if command[2] == "upload":
                raise subprocess.CalledProcessError(1, "gh release upload")

        with (
            patch.object(release, "api", side_effect=[{"sha": SHA}, None, None]),
            patch.object(release, "run", side_effect=gh) as run,
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                release.create(self.release_args())
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
                    patch.object(release, "api", side_effect=replies),
                    patch.object(release, "run") as run,
                ):
                    with self.assertRaises(ValueError):
                        release.create(self.release_args())
                    run.assert_not_called()

    def test_delete_includes_tag_and_only_skips_prompt_with_yes(self):
        for yes in (False, True):
            with (
                patch.object(release, "run") as run,
                patch.object(release, "api", return_value={"object": {}}),
            ):
                release.delete(
                    argparse.Namespace(
                        version="2026-10-08", repo=release.indexes.REPO, yes=yes
                    )
                )
            self.assertIn("--cleanup-tag", run.call_args.args)
            self.assertEqual("--yes" in run.call_args.args, yes)

    def test_delete_draft_without_a_tag(self):
        with (
            patch.object(release, "run") as run,
            patch.object(release, "api", return_value=None),
        ):
            release.delete(
                argparse.Namespace(
                    version="2026-10-08", repo=release.indexes.REPO, yes=True
                )
            )
        self.assertEqual(run.call_args.args[2], "delete")
        self.assertNotIn("--cleanup-tag", run.call_args.args)


class ArgumentsTest(unittest.TestCase):
    def test_cli_routes_create_and_delete_to_release_operations(self):
        for command in ("create", "delete"):
            with (
                patch.object(
                    sys, "argv", ["release", command, "--version", "2026-10-08"]
                ),
                patch.object(release, "create") as create,
                patch.object(release, "delete") as delete,
            ):
                release.main()
                action = create if command == "create" else delete
                action.assert_called_once()
                self.assertEqual(action.call_args.args[0].version, "2026-10-08")

    def test_required_versions_and_invalid_dates(self):
        for args in (
            ("create",),
            ("delete",),
            ("delete", "--version", "2026-02-30"),
        ):
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/release/cli.py"), *args],
                capture_output=True,
            )
            self.assertEqual(result.returncode, 2, args)

    def test_api_does_not_hide_authentication_or_server_errors(self):
        for code in (401, 403, 500):
            result = subprocess.CompletedProcess([], 1, "", f"gh: failed (HTTP {code})")
            with patch.object(release.subprocess, "run", return_value=result):
                with self.assertRaises(RuntimeError):
                    release.api(release.indexes.REPO, "x", optional=True)
        missing = subprocess.CompletedProcess([], 1, "", "gh: Not Found (HTTP 404)")
        with patch.object(release.subprocess, "run", return_value=missing):
            self.assertIsNone(release.api(release.indexes.REPO, "x", optional=True))


if __name__ == "__main__":
    unittest.main()
