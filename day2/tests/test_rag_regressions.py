"""CodeRabbit 指摘の回帰テスト。OpenAI / Weave への通信は行わない。

uv run python -m unittest discover -s tests -v
"""

import json
import os
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["WANDB_PROJECT"] = "test-project"
os.environ["WEAVE_DISABLED"] = "true"

from app.advanced_rag import build_index, ingest, rag  # noqa: E402
from app.advanced_rag.extractors import render_drawing, slides, vision  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]


class DrawingTest(unittest.TestCase):
    def test_missing_soffice_can_be_retried(self):
        data = b"\xd7\xcd\xc6\x9a" + b"test metafile"
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(render_drawing, "METAFILE_CACHE", Path(tmp)),
            patch.object(render_drawing.subprocess, "run") as run,
        ):
            run.side_effect = FileNotFoundError("soffice")
            self.assertIsNone(render_drawing.metafile_png(data))
            self.assertEqual(list(Path(tmp).iterdir()), [])

            def convert(args, **kwargs):
                out = Path(args[args.index("--outdir") + 1]) / "metafile.png"
                Image.new("RGB", (10, 10), "black").save(out)

            run.side_effect = convert
            png = render_drawing.metafile_png(data)
            self.assertTrue(png.startswith(b"\x89PNG"))
            self.assertEqual(render_drawing.metafile_png(data), png)
            self.assertEqual(run.call_count, 2)

    def test_conversion_failure_is_still_cached(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(render_drawing, "METAFILE_CACHE", Path(tmp)),
            patch.object(
                render_drawing.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "soffice"),
            ) as run,
        ):
            for _ in range(2):
                self.assertIsNone(render_drawing.metafile_png(b"\xd7\xcd\xc6\x9a"))
            run.assert_called_once()
            self.assertEqual(next(Path(tmp).iterdir()).read_bytes(), b"")

    def test_absolute_and_cell_anchors(self):
        ns = render_drawing.NS["xdr"]
        start = "<xdr:from><xdr:col>1</xdr:col><xdr:colOff>9525</xdr:colOff><xdr:row>1</xdr:row><xdr:rowOff>19050</xdr:rowOff></xdr:from>"
        extent = '<xdr:ext cx="285750" cy="381000"/>'
        cases = [
            (
                "absoluteAnchor",
                '<xdr:pos x="95250" y="190500"/>' + extent,
                [10, 20, 30, 40],
            ),
            ("oneCellAnchor", start + extent, [101, 202, 30, 40]),
            (
                "twoCellAnchor",
                start
                + "<xdr:to><xdr:col>2</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>2</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:to>",
                [101, 202, 99, 198],
            ),
        ]
        for tag, body, expected in cases:
            with self.subTest(tag=tag):
                anchor = ET.fromstring(
                    f'<xdr:{tag} xmlns:xdr="{ns}">{body}</xdr:{tag}>'
                )
                self.assertEqual(
                    render_drawing.anchor_box(anchor, [0, 100, 200], [0, 200, 400]),
                    expected,
                )


class SlideTest(unittest.TestCase):
    def test_descriptions_without_text_and_after_last_line(self):
        line = ((0, 0, 10, 10), ("本文", "heading"))
        descriptions = json.dumps(
            {
                "items": [
                    {"after": 99, "text": "末尾の説明"},
                    {"after": 1, "text": "行の説明"},
                    {"after": 0, "text": "別の末尾説明"},
                ]
            }
        )
        with (
            patch.object(
                slides,
                "Presentation",
                return_value=SimpleNamespace(slides=[("空", "a"), ("本文あり", "b")]),
            ),
            patch.object(
                slides, "render_slides", return_value={"空": None, "本文あり": None}
            ),
            patch.object(slides, "collect_slide"),
            patch.object(slides, "boxed_lines", side_effect=[[], [line]]),
            patch.object(
                vision,
                "describe",
                return_value={"空": descriptions, "本文あり": descriptions},
            ),
        ):
            sections = slides.sections_vision(Path("test.pptx"))
        self.assertEqual(
            sections[0][1],
            vision.description_lines("末尾の説明\n行の説明\n別の末尾説明"),
        )
        self.assertEqual(
            sections[1][1],
            [line[1]] + vision.description_lines("行の説明\n末尾の説明\n別の末尾説明"),
        )


class IndexTest(unittest.TestCase):
    def test_utf8_round_trip_and_quoted_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "sample.xlsx").touch()
            collections = ["O'Reilly資料📚", "x' OR 1=1 --", "別の資料"]
            records = [
                {
                    "id": str(i),
                    "title": "検索📚",
                    "text": "顧客検索の説明📚",
                    "collection": collection,
                    "file": f"{collection}/仕様.xlsx",
                    "sheet": "表'1",
                    "unit": "表'1",
                    "seq": 0,
                }
                for i, collection in enumerate(collections)
            ]
            original_open = Path.open

            def checked_open(path, mode="r", *args, **kwargs):
                if path.suffix in (".jsonl", ".json") and "b" not in mode:
                    self.assertEqual(kwargs.get("encoding"), "utf-8")
                return original_open(path, mode, *args, **kwargs)

            with (
                patch.object(rag, "DB_DIR", root / "db"),
                patch.object(build_index, "DB_DIR", root / "db"),
                patch.object(ingest, "CORPUS_DIR", corpus),
                patch.object(ingest, "ingest_file", return_value=records),
                patch.object(
                    build_index, "embed", return_value=np.array([[1.0, 0.0]] * 3)
                ),
                patch.object(rag, "embed", return_value=np.array([[1.0, 0.0]])),
                patch.object(rag, "_tables", {}),
                patch.object(Path, "open", checked_open),
            ):
                build_index.build("structured", "test-embedding")
                self.assertEqual(rag.load_jsonl(rag.docs_path("structured")), records)
                self.assertEqual(
                    rag.index_meta("structured")["embedding_model"], "test-embedding"
                )
                for search in ("vector", "fts", "hybrid"):
                    model = rag.RagModel(
                        search=search,
                        use_collection=True,
                        embedding_model="test-embedding",
                    )
                    for i, collection in enumerate(collections):
                        with self.subTest(search=search, collection=collection):
                            result = model.retrieve("顧客検索", collection)
                            self.assertEqual(result["ranked_ids"], [str(i)])
                    model.expand = "unit"
                    self.assertEqual(
                        model.retrieve("顧客検索", collections[0])["contexts"][0]["id"],
                        "0",
                    )
                    model.expand = "none"
                    model.use_collection = False
                    self.assertEqual(
                        len(model.retrieve("顧客検索", collections[0])["contexts"]), 3
                    )

    def test_vision_cache_uses_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.pptx"
            source.write_bytes(b"slide")
            images = {"slide": Image.new("RGB", (10, 10), "black")}
            with (
                patch.object(vision, "CACHE_DIR", root / "cache"),
                patch.object(
                    vision, "_describe", return_value="図の説明📚"
                ) as describe,
            ):
                self.assertEqual(
                    vision.describe(source, images, "prompt"), {"slide": "図の説明📚"}
                )
                self.assertEqual(
                    vision.describe(source, images, "prompt"), {"slide": "図の説明📚"}
                )
                describe.assert_called_once()
            self.assertEqual(
                next((root / "cache").iterdir()).read_bytes(),
                "図の説明📚".encode("utf-8"),
            )


class PageTest(unittest.TestCase):
    @patch("weave.init")
    def test_simple_rag_resets_on_page_change_only(self, init):
        app = AppTest.from_file(str(PROJECT / "pages/part2_1_rag.py"))
        app.session_state["current_page"] = "other"
        app.session_state["call"] = "incompatible other page state"
        app.session_state["output"] = "incompatible other page state"
        app.run()
        self.assertEqual(list(app.exception), [])
        self.assertNotIn("call", app.session_state)
        self.assertNotIn("output", app.session_state)
        app.session_state["marker"] = "same page"
        app.run()
        self.assertEqual(app.session_state["marker"], "same page")
        self.assertEqual(list(app.exception), [])

    @patch("weave.init")
    def test_downloads_reject_missing_and_outside_files_and_allow_duplicate_ids(
        self, init
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "sample.txt").write_text("content", encoding="utf-8")
            outside = root / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            (corpus / "link.txt").symlink_to(outside)
            files = [
                "sample.txt",
                "sample.txt",
                "missing.txt",
                "../outside.txt",
                str(outside),
                "link.txt",
                ".",
            ]
            contexts = [
                {"id": "duplicate", "title": "title", "text": "text", "file": f}
                for f in files
            ]
            script = f"""import runpy
from pathlib import Path
page = runpy.run_path({str(PROJECT / "pages/part3_1_advanced_rag.py")!r})
model_class = page["GuiRagModel"]
model_class.retrieve.__globals__["CORPUS_DIR"] = Path({str(corpus)!r})
model_class().retrieve("question")
"""
            with patch.object(
                rag.RagModel,
                "retrieve",
                return_value={"contexts": contexts, "ranked_ids": []},
            ):
                app = AppTest.from_string(script).run()
            self.assertEqual(list(app.exception), [])
            buttons = app.get("download_button")
            self.assertEqual(len(buttons), 2)
            self.assertNotEqual(buttons[0].proto.id, buttons[1].proto.id)
            self.assertEqual(
                sum("ダウンロードできません" in c.value for c in app.caption), 5
            )


if __name__ == "__main__":
    unittest.main()
