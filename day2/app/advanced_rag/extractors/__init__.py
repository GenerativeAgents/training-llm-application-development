"""オフィスファイル(xlsx / pdf / pptx)からテキストを取り出す方式(抽出器)の切り替え。

抽出器は `extract(path) -> list[Chunked]` の関数。Chunked は位置単位 1 つ分の (区切り名, チャンクの本文リスト) で、
長い位置単位をどうチャンクに切るかも抽出器の責任(共通の切り方は chunking.py)。
抽出器の内部では、切る前の Section = (区切り名, 行のリスト) を組み立てる(vision はこれに図の説明を足す)。
区切りは、xlsx はシート、pptx はスライド(`slide08`)、pdf はページ(`page05`。structured / vision はしおりの項目で、区切り名はページ範囲 `page18-21`)が 1 区切り。
行は (テキスト, 種別) で、種別は "heading"(見出し) / "header"(表のヘッダ行) / ""。
種別は分割時に、チャンクの先頭へ見出し・ヘッダを付け直すために使う(無ければ "" でよい)。
チャンクの ID(`相対パス#区切り名#連番`)は ingest.py が付けるので、抽出器は区切り名を正解データの粒度に合わせるだけでよい。

抽出器は研修の段階(README の「段階」)に合わせて、手をかける順に残してある:

    uv run python -m app.advanced_rag.ingest --extractor basic        # 段階 1, 2: 汎用ツールの素のテキスト(markitdown。PDF は中身の pdfminer)を位置単位で分けるだけ
    uv run python -m app.advanced_rag.ingest --extractor structured   # 段階 3: 文書の構造に合わせて抽出・チャンキング(不要な情報の除去、ヘッダの繰り返しなど)。Vision は使わない
    uv run python -m app.advanced_rag.ingest --extractor vision       # 段階 4: structured + 図形・画像をマルチモーダル LLM に説明させる
"""

from collections.abc import Callable
from pathlib import Path

Line = tuple[str, str]
Section = tuple[str, list[Line]]  # 切る前: (区切り名, 行のリスト)
# 切った後: (区切り名, チャンクの本文リスト)。3 つ目に、チャンクごとの付帯情報(dict)のリストを付けてもよい:
# - unit: そのチャンクを切り出した区切り。無ければ区切り名。PDF はしおりの項目の区切り名がチャンクのページ範囲と
#   違うので、項目全体のページ範囲を入れる(回答時に同じ区切りのほかのチャンクを渡すのに使う。段階 8)
Chunked = tuple[str, list[str]] | tuple[str, list[str], list[dict]]
Extractor = Callable[[Path], list[Chunked]]

from . import basic, chunking, pdf, slides, structured  # noqa: E402  (上の定義を参照するため後置)


def _vision(path: Path) -> list[Chunked]:
    from . import vision  # openai と Pillow が要るので、使うときだけ読み込む

    return vision.extract(path)


def _by_suffix(xlsx: Extractor, pdf_: Extractor, pptx: Extractor) -> Extractor:
    """拡張子で xlsx / pdf / pptx 用の抽出器を使い分ける。抽出器の意味は同じ(pdf.py / slides.py の説明を参照)。"""

    def extract(path: Path) -> list[Chunked]:
        return {".pdf": pdf_, ".pptx": pptx}.get(path.suffix, xlsx)(path)

    return extract


EXTRACTORS: dict[str, Extractor] = {
    "basic": basic.extract,  # 形式の振り分けは basic.py の中
    "structured": _by_suffix(structured.extract, pdf.extract, slides.extract),
    "vision": _by_suffix(_vision, pdf.extract_vision, slides.extract_vision),
}
SUFFIXES = {".xlsx", ".pdf", ".pptx"}
