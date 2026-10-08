"""段階 1, 2 で使う: 汎用ツールの素のテキストを、位置単位(シート / スライド / ページ)で分けるだけの抽出器。

最も原始的な RAG の比較用。テキストには手を加えず、長い位置単位は共通の切り方(chunking.split_chunks)で文字数で切る。
位置単位は評価の正解データ(source_ids)の粒度と同じで、ID は `相対パス#シート名` / `#slide01` / `#page01`。

- xlsx: markitdown。シートごとに "## シート名" 見出し + Markdown 表(pandas 経由)になる。
  帳票レイアウトだと空セルが NaN / Unnamed として大量に残るが、定型シート(表紙など)も含めてそのまま使う
- pptx: markitdown。スライドごとに "<!-- Slide number: N -->" が入る。図形は XML の出現順(= 重なり順)
- pdf: markitdown はページの区切りを出さないので、その中身と同じ pdfminer.six を直接使う。
  pdfminer はページごとに改ページ文字(\\f)を出す
"""

import re
import warnings
from pathlib import Path

from markitdown import MarkItDown
from pdfminer.high_level import extract_text

from . import Chunked, Section
from .chunking import chunk_sections

warnings.filterwarnings("ignore")  # pandas / openpyxl の警告

SLIDE_MARK_RE = re.compile(r"^<!-- Slide number: (\d+) -->$", re.M)

_md = MarkItDown()


def _lines(text: str) -> list[tuple[str, str]]:
    return [(line.rstrip(), "") for line in text.splitlines() if line.strip()]


def _xlsx(path: Path) -> list[Section]:
    text = _md.convert(str(path)).text_content
    sections = []
    for block in re.split(r"^(?=## )", text, flags=re.M):  # 最初の "## " より前には何も無い
        if block.startswith("## "):
            title, _, body = block.partition("\n")
            sections.append((title[3:].strip(), _lines(body)))
    return sections


def _pptx(path: Path) -> list[Section]:
    text = _md.convert(str(path)).text_content
    parts = SLIDE_MARK_RE.split(text)  # [前置き, 番号, 本文, 番号, 本文, ...]
    return [(f"slide{int(n):02d}", _lines(body.split("### Notes:")[0])) for n, body in zip(parts[1::2], parts[2::2])]


def _pdf(path: Path) -> list[Section]:
    pages = extract_text(path).split("\f")  # 末尾は空
    return [(f"page{i:02d}", _lines(text)) for i, text in enumerate(pages, 1) if text.strip()]


def extract(path: Path) -> list[Chunked]:
    return chunk_sections({".pdf": _pdf, ".pptx": _pptx}.get(path.suffix, _xlsx)(path))
