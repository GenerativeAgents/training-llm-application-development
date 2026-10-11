"""PDF の抽出器。

抽出器の考え方は xlsx と同じ(basic は basic.py で、ページ単位):

- structured(extract): PDF のしおり(アウトライン)が 2 階層以上あれば、しおりの項目ごとに区切る
  (操作手順書の `1.3.3 パスワードの設定` など)。手順がページをまたいでも 1 つのチャンクに入る。
  チャンクの区切り名はそのチャンクが含むページの範囲(`page18-21`。1 ページなら `page18`)にして、
  ページ単位の正解データ(`#page18`)と照合できるようにする(notebooks/part3_1_eval.ipynb の covers)。
  チャンクの先頭にはしおりの階層(`第1章 … > 1.3 ファイルの保存 > 1.3.3 パスワードの設定`)を見出しとして付け、
  目次と、しおりの題と同じ行(章名の柱)は落とす。
  しおりが無い、または 1 階層だけ(厚労省資料の「スライド N」)の PDF は、basic と同じくページで区切る。
  テキストは basic と同じ pdfminer.six で、ページごとに取る。どちらの区切り方でも、ページの上端・下端の帯にある
  ページ番号(数字と記号だけの行)と、奇数または偶数ページの半分以上に同じ文字で出る行(柱: 資料名など)を落とす。
  資料ごとの形は決め打ちしない(担当課のように一部のページにだけ出る行は内容なので残る)
- vision(extract_vision): structured と同じ区切りに、マルチモーダル LLM の説明を `図の説明:` 行として本文の中に差し込む。
  ページごとに 1 回、ページ全体を描いた画像と、テキスト層の行(番号と画像上の座標付き)を一緒に渡す。
  - 画像で潰れた文字は同じ位置のテキスト行で確かめられ、箱のラベルと矢印の対応も座標で確定しやすい。
    画面キャプチャなど貼り付けられた画像の中の文字はテキスト層に無いので、画像から読ませる
  - 貼り付け画像を切り出さないので、画像の外にある番号や引き出し線も一緒に読める
  - 説明は vision.Figure の JSON(矢印・列の対応・その他の事実)で返させ、行の文面は vision.figure_lines が組み立てる。
    説明ごとに「どのテキスト行の後ろに入れるか」を行番号(after)で返させ、その行の後ろに差し込む
    (手順の行と、その手順の画面の説明が並ぶ)。図も画像も無いページは空で返させ、差し込まない
  - PDF には図形の情報が無いので、矢印の向きは LLM が画像から読む(xlsx / pptx は XML から確定する。arrows.py)。
    ただし箱の列の対応(「A の下は B」)は、ベクタ描画の矩形と中の文字から確定できる(vector_pairs)ので、LLM に任せず足す
  全ページを読ませる(図のあるページを図形の数で選ぶ方法は、手順書にも枠や角丸の囲みが多く、資料によらずには決められない)

章扉などの太字が文字の二重打ちで描かれていると、文字が二重になる(`第第11章章`)ので正規化する。
"""

import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextLine
from pdfminer.pdfdocument import PDFDocument, PDFNoOutlines
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import resolve1

from . import Chunked, ChunkedWithMeta
from .chunking import split_chunks_with_index
from .figure_rules import OMIT, RULES, SCHEMA

HEADING_RE = re.compile(r"^(?:\d+|[A-Z])\.\d+\.\d+ \S")  # 1.3.1 基本ファイル形式での保存
DOUBLED_RE = re.compile(r"^(?:(.)\1|\s)+$")  # 章扉の太字: 文字が 2 つずつ重なる
TOC_RE = re.compile(r"\.{5}|・{5}|…{3}")  # 目次の点線(節の形をしているが見出しではない)
MARGIN = 0.08  # ページの上端・下端のこの割合の帯を、ヘッダ/フッタの置き場とみなす
PAGE_NUMBER_RE = re.compile(r"[A-Za-z]{0,4}[-‐–—.．/／()（）\[\]・]{0,4}")  # 数字を除いた残りがこれだけならページ番号("1-3", "A-2", "- 5 -", "p.5")


def _undouble(line: str) -> str:
    return re.sub(r"(.)\1", r"\1", line)


# --- structured: しおり単位の区切り -------------------------------------------------------------


def outline(path: Path) -> list[tuple[int, str, int]]:
    """しおりを (階層, 題, ページ番号) の出現順で返す。しおりが無ければ空(呼び出し側はページで区切る)。

    行き先がページに解決できない項目(URL へのリンク、壊れた参照など)は飛ばす。
    """
    with path.open("rb") as f:
        doc = PDFDocument(PDFParser(f))
        pages = {page.pageid: i for i, page in enumerate(PDFPage.create_pages(doc), 1)}
        try:
            items = list(doc.get_outlines())
        except PDFNoOutlines:
            return []
        result = []
        for level, title, dest, action, _ in items:
            try:
                dest = resolve1(dest) if dest else None
                if dest is None and action:
                    dest = resolve1(resolve1(action).get("D"))
                if isinstance(dest, (bytes, str)):  # 名前付きの行き先
                    dest = resolve1(doc.get_dest(dest))
                if isinstance(dest, dict):
                    dest = dest.get("D")
                pno = pages.get(getattr(dest[0], "objid", None)) if dest else None
            except Exception:  # noqa: BLE001  壊れたしおりは飛ばす(抽出全体は止めない)
                pno = None
            if pno:
                result.append((level, str(title).strip(), pno))
        return result


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def page_texts(path: Path) -> list[list[str]]:
    """ページごとの行(空行を除く)。basic と同じく pdfminer の改ページ文字で分ける。

    pdfminer は各ページの最後に改ページ文字を出すので、末尾に空のページが 1 つ余分に付く(ページ数 + 1 要素)。
    _outline_units は最後の項目の終端にこの空ページを使う。
    """
    return [[line.strip() for line in text.splitlines() if line.strip()] for text in extract_text(path).split("\f")]


def running_lines(path: Path) -> list[set[str]]:
    """ページごとの、落とすヘッダ/フッタの行の文字。

    ページの上端・下端(MARGIN)の帯にある行のうち、次のどちらかに当たるもの:
    - ページ番号: 数字を # にすると数字と記号(と短い英字)だけが残り、同じ形が 3 ページ以上にある
    - 柱: 同じ文字が、奇数ページか偶数ページの半分以上にある(見開きで左右の柱が違う体裁があるので分けて数える)
    行の位置は pdfminer のレイアウトから取る(page_texts と同じ解析なので行の文字は一致する)。
    """
    bands: list[set[str]] = []
    for page in extract_pages(path):
        found: set[str] = set()
        stack = list(page)
        while stack:
            el = stack.pop()
            if isinstance(el, LTTextLine):
                if el.y1 > page.height * (1 - MARGIN) or el.y0 < page.height * MARGIN:
                    found.update(t.strip() for t in el.get_text().splitlines() if t.strip())
            elif hasattr(el, "__iter__"):
                stack.extend(el)
        bands.append(found)

    def number_shape(line: str) -> str | None:
        key = re.sub(r"[0-9０-９]+", "#", _squash(line))
        return key if "#" in key and PAGE_NUMBER_RE.fullmatch(key.replace("#", "")) else None

    shapes = Counter(k for band in bands for k in {number_shape(t) for t in band} if k)
    running = set()
    for parity in (0, 1):
        pages = [band for i, band in enumerate(bands, 1) if i % 2 == parity]
        counts = Counter(t for band in pages for t in band)
        running |= {t for t, c in counts.items() if c >= 3 and c >= len(pages) / 2}
    return [{t for t in band if t in running or shapes.get(cast(str, number_shape(t)), 0) >= 3} for band in bands]


def _range_name(pages: list[int]) -> str:
    first, last = min(pages), max(pages)
    return f"page{first:02d}" if first == last else f"page{first:02d}-{last:02d}"


def _chunk_with_pages(units: list[list[tuple[str, str, int]]]) -> list[Chunked]:
    """区切りごとの (行, 種別, ページ) をチャンクに切り、チャンクが含むページの範囲を区切り名にする。

    同じ範囲名のチャンクが続けば 1 つにまとめる(ingest.py が #1, #2 を付ける)。
    チャンクごとの unit は、そのチャンクを切り出した区切り(しおりの項目、またはページ)全体のページ範囲
    (回答時に、チャンクの範囲ではなく区切りの境界で前後のチャンクを渡すため)。
    """
    result: list[Chunked] = []
    for lines in units:
        meta = {"unit": _range_name(sorted({p for _, _, p in lines}))}
        for text, index in split_chunks_with_index([(t, k) for t, k, _ in lines]):
            name = _range_name([lines[i][2] for i in index])
            if result and result[-1][0] == name:
                result[-1][1].append(text)
                cast(ChunkedWithMeta, result[-1])[2].append(meta)
            else:
                result.append((name, [text], [meta]))
    return result


def _outline_units(pages: list[list[str]], items: list[tuple[int, str, int]]) -> list[list[tuple[str, str, int]]]:
    """しおりの項目ごとに (行, 種別, ページ) を集める。先頭にしおりの階層を見出し行として置く。

    項目の題と同じ行がそのページにあれば、そこで区切る(同じページで次の項目が始まる場合)。
    無ければページの先頭で区切る。最初の項目より前(表紙など)は捨てる。
    """
    chapters = {_squash(re.sub(r"^\S+ ", "", title)) for level, title, _ in items if level == 1}
    starts: list[tuple[int, int]] = []  # 項目ごとの (ページ, 行番号)
    for _, title, pno in items:
        lines = pages[pno - 1]
        begin = starts[-1][1] + 1 if starts and starts[-1][0] == pno else 0
        found = next((i for i in range(begin, len(lines)) if _squash(lines[i]).startswith(_squash(title))), None)
        starts.append((pno, found if found is not None else begin))
    units = []
    path: list[str] = []  # 今の項目までのしおりの題(階層ごと)
    for k, (level, title, _) in enumerate(items):
        path = path[: level - 1] + [title]
        start = starts[k]
        # 最後の項目の終端は pages 末尾の空ページ(page_texts 参照)の先頭。本文のある最後のページまで全部入る
        end = starts[k + 1] if k + 1 < len(items) else (len(pages), 0)
        body = []
        for pno in range(start[0], end[0] + 1):
            lines = pages[pno - 1] if pno <= len(pages) else []
            lo = start[1] if pno == start[0] else 0
            hi = end[1] if pno == end[0] else len(lines)
            for i in range(lo, hi):
                line = lines[i]
                if i == start[1] and pno == start[0] and _squash(line).startswith(_squash(title)):
                    continue  # 項目の題そのもの(階層の見出し行に含めた)
                if _squash(line) in chapters or TOC_RE.search(line):
                    continue
                if len(line) >= 4 and DOUBLED_RE.match(line):
                    line = _undouble(line)
                body.append((line, "heading" if HEADING_RE.match(line) else "", pno))
        if body:
            units.append([(" > ".join(path), "heading", start[0])] + body)
    return units


def units(path: Path) -> list[list[tuple[str, str, int]]]:
    """区切りごとの (行, 種別, ページ)。しおりが 2 階層以上ならしおりの項目ごと、それ以外はページごと。"""
    drop = running_lines(path)
    pages = [[line for line in lines if line not in (drop[i] if i < len(drop) else ())] for i, lines in enumerate(page_texts(path))]
    items = outline(path)
    if max((level for level, _, _ in items), default=0) >= 2:
        return _outline_units(pages, items)
    # しおりが無い / 1 階層だけ: ページで区切る
    return [[(line, "", i) for line in lines] for i, lines in enumerate(pages, 1) if lines]


def extract(path: Path) -> list[Chunked]:
    return _chunk_with_pages(units(path))


# --- vision -------------------------------------------------------------------------------------

PAGE_PROMPT = f"""これは PDF 資料の 1 ページを画像にしたものと、そのページのテキスト層から取り出した行です。
テキスト行は `L番号 [x0,top,x1,bottom] 文字` の形で、座標は画像の画素です。
本文の文章は別に取り出してあるので、書き写さないでください。ページ上の図と画像の内容を、検索で引けるように説明してください。

文字の読み方:
- 画像に見えている文字は、同じ位置にあるテキスト行の文字を正とする(画像では潰れて読みにくい文字も、テキスト行で確かめる)
- テキスト行にあっても画像に見えない文字(隠れた文字)は使わない
- 画面キャプチャなど貼り付けられた画像の中の文字はテキスト行に無いので、画像から読む

書くこと:
- 画面キャプチャは、先頭にどの手順・見出しの画面か(例:「手順 3 の画面」)を facts に書く
{RULES}

{OMIT}

{SCHEMA}
after は、その説明を本文のどの行の後ろに差し込むかで、説明している図や画面の直前にある、その手順や見出しの行の番号(L の後ろの数字)にします。"""


PAIR_GAP_PT = 80  # 上下の箱の列(pairs)とみなす、上の段の下端から下の段の上端までの距離の上限(pt)
PAIR_TEXT_CHARS = 30  # 箱の文字数の上限。これより長いのは文の入った枠で、列の対応の箱ではない


def _box_text(words: list[dict[str, Any]], rect: dict[str, Any]) -> str:
    """矩形の中にある語を読み順につなぐ(日本語は空白なし、英数字どうしは空白)。"""
    inside = sorted((w for w in words if rect["x0"] <= (w["x0"] + w["x1"]) / 2 <= rect["x1"]
                     and rect["top"] <= (w["top"] + w["bottom"]) / 2 <= rect["bottom"]), key=lambda w: (round(w["top"]), w["x0"]))
    text = ""
    for w in inside:
        text += (" " if text and text[-1].isascii() and w["text"][:1].isascii() else "") + w["text"]
    return text


def vector_pairs(page: Any) -> list[tuple[str, str]]:
    """ページのベクタ描画から、箱の列の上下の対応 (上の箱の文字, 下の箱の文字) を取る。

    文字の入った枠線付きの矩形を箱とし、縦の範囲が重なる箱を段にまとめる。段が 2 つ縦に並び(間隔 PAIR_GAP_PT 以内)、
    どちらも 2 箱以上で数が同じなら、横の位置が重なる箱どうしを対応とする(「本社 | 支社」の下に
    「東日本 | 西日本」が並ぶポンチ絵の形)。3 段以上続く表のセルは対象にしない。
    """
    words = page.extract_words()
    rects = [r for r in page.rects if r.get("stroke") and 20 <= r["width"] <= 300 and 8 <= r["height"] <= 100]
    boxes = []
    for r in rects:
        if any(o is not r and r["x0"] <= o["x0"] and o["x1"] <= r["x1"] and r["top"] <= o["top"] and o["bottom"] <= r["bottom"]
               and (o["width"] < r["width"] - 1 or o["height"] < r["height"] - 1) for o in rects):
            continue  # ほかの箱を含む枠(大枠)
        text = _box_text(words, r)
        if text and not any(b["x0"] == r["x0"] and b["top"] == r["top"] for b in boxes):
            boxes.append({**r, "text": text})
    rows: list[list[dict[str, Any]]] = []
    for b in sorted(boxes, key=lambda b: (b["top"], b["x0"])):
        row = next((row for row in rows if min(row[0]["bottom"], b["bottom"]) - max(row[0]["top"], b["top"]) > b["height"] / 2), None)
        if row is None:
            rows.append([b])
        else:
            row.append(b)
    rows.sort(key=lambda row: row[0]["top"])
    pairs = []
    for i, (upper, lower) in enumerate(zip(rows, rows[1:])):
        if len(upper) < 2 or len(upper) != len(lower) or lower[0]["top"] - max(b["bottom"] for b in upper) > PAIR_GAP_PT:
            continue
        stacked = lambda a, b: len(a) == len(b) and b[0]["top"] - max(x["bottom"] for x in a) <= PAIR_GAP_PT  # noqa: E731
        if (i > 0 and stacked(rows[i - 1], upper)) or (i + 2 < len(rows) and stacked(lower, rows[i + 2])):
            continue  # 3 段以上続く(表)
        for u in upper:
            for lo in lower:
                if (u["x0"] <= (lo["x0"] + lo["x1"]) / 2 <= u["x1"] and max(len(u["text"]), len(lo["text"])) <= PAIR_TEXT_CHARS
                        and max(u["width"], lo["width"]) <= 2 * min(u["width"], lo["width"])):  # 文が入った枠や幅の違う帯は箱ではない
                    pairs.append((u["text"], lo["text"]))
    return pairs


def page_inputs(path: Path) -> tuple[dict[str, bytes], dict[str, list[tuple[list[int], str]]], dict[str, list[tuple[str, str]]]]:
    """ページ全体の画像(`p15` → PNG のバイト列)、テキスト層の行(`p15` → (座標, 文字)のリスト)、箱の列の対応(`p15` → vector_pairs)を返す。

    画像は長い辺が vision.MAX_SIDE になる解像度で描く(送るときに縮小されない大きさ)。
    数百ページの PDF があるので、描いた画像は Pillow のまま持たずにすぐ PNG にし(1 ページ 8MB 超 → 数百 KB)、
    pdfplumber がページごとに溜める解析結果も捨てる(flush_cache)。8GB のマシンでも全ページを処理できるようにするため。
    行は pdfplumber の extract_text_lines の順で、プロンプトの `L番号` はこのリストの 1 始まりの番号。
    """
    import pdfplumber

    from .vision import MAX_SIDE, png_bytes

    images, lines, pairs = {}, {}, {}
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, 1):
            scale = MAX_SIDE / max(page.width, page.height)
            images[f"p{pno}"] = png_bytes(page.to_image(resolution=72 * scale).original)
            lines[f"p{pno}"] = [
                ([round(tl[k] * scale) for k in ("x0", "top", "x1", "bottom")], tl["text"].strip())
                for tl in page.extract_text_lines()
            ]
            pairs[f"p{pno}"] = vector_pairs(page)
            page.flush_cache()
    return images, lines, pairs


def extract_vision(path: Path) -> list[Chunked]:
    from . import vision  # 読み込むと OpenAI のクライアントを作る(OPENAI_API_KEY が要る)ので、使うときだけ読み込む

    images, lines, pairs = page_inputs(path)
    extra = {key: "\n\n# テキスト行\n" + "\n".join(f"L{n} {box} {text}" for n, (box, text) in enumerate(page, 1))
             for key, page in lines.items()}
    described = vision.describe(path, images, PAGE_PROMPT, extra=extra, schema=vision.Figure)
    result = units(path)
    for key in sorted(described, key=lambda k: int(k[1:])):
        pno, page = int(key[1:]), lines[key]
        cursor = None  # そのページで前回差し込んだ説明の末尾の位置
        # 差し込む行ごとにまとめ、ページの上から順に差し込む(行番号が範囲外なら、ページの最後の行の後ろ)
        groups: dict[int, list[str]] = {}
        described_lines = vision.figure_lines(described[key])
        for after, text in described_lines:
            groups.setdefault(after if 1 <= after <= len(page) else len(page) + 1, []).append(text)
        # 箱の列の対応は矩形から確定したものを足す(LLM が pairs に入れていなければ)。差し込み先は上の箱の文字と同じテキスト行
        for upper, lower in pairs.get(key, []):
            text = f"{vision.PREFIX}{upper} の下は {lower}"
            if any(_squash(text) == _squash(t) for _, t in described_lines):
                continue
            at = next((n for n, (_, t) in enumerate(page, 1) if _squash(t) and _squash(upper).startswith(_squash(t))), len(page) + 1)
            groups.setdefault(at, []).append(text)
        for after in sorted(groups):
            anchor = page[after - 1][1] if after <= len(page) else ""
            cursor = _insert(result, pno, anchor, [(text, "") for text in groups[after]], cursor)
    return _chunk_with_pages(result)


def _matches(line: str, key: str) -> bool:
    """pdfminer の行(line)が、pdfplumber の行の先頭 20 字(key)と同じ行か。行の折り返し位置が違うことがあるので両向きに見る。"""
    return line.startswith(key) or (len(line) >= 6 and key.startswith(line))


def _insert(result: list[list[tuple[str, str, int]]], pno: int, anchor: str, desc: list[tuple[str, str]],
            after: tuple[int, int] | None = None) -> tuple[int, int] | None:
    """ページ pno の、anchor で始まる行の後ろ(先に差し込んだ説明があればその後ろ)に説明を差し込む。

    anchor は LLM が指定したテキスト行(pdfplumber)の文字。本文の行(pdfminer)とは分け方が違うことがあるので、
    文字の先頭で照合する。anchor が空か見つからなければ、そのページの最後の行の後ろ。
    同じ文の行が繰り返されても取り違えないよう、anchor は after(そのページで前回差し込んだ位置)より後ろから探す。
    差し込んだ説明の末尾の位置を返す。ページの行がどの区切りにも無い(文字の無いページ、しおりの前の表紙)なら捨てる。
    """
    spots = [(u, i) for u, lines in enumerate(result) for i, (_, _, p) in enumerate(lines) if p == pno]
    if not spots:
        return after
    at = spots[-1]
    key = _squash(anchor)[:20]
    if key:
        found = [(u, i) for u, i in spots
                 if (after is None or (u, i) >= after) and not result[u][i][0].startswith("図の説明: ")
                 and _matches(_squash(result[u][i][0]), key)]
        if found:
            at = found[0]
    u, i = at
    lines = result[u]
    i += 1
    while i < len(lines) and lines[i][0].startswith("図の説明: ") and lines[i][2] == pno:
        i += 1
    lines[i:i] = [(text, kind, pno) for text, kind in desc]
    return (u, i + len(desc))
