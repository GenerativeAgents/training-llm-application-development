"""structured 抽出器に、図や画像の「図の説明」をマルチモーダル LLM で足す抽出器(段階 4)。

structured は図形の文字だけを出現順に並べるので、矢印の向きやチェックボックスの状態が失われる。
貼り付けられた画像(画面キャプチャ)の中の文字も取れない。ここでは次の 2 つをマルチモーダル LLM に説明させ、
`図の説明:` 行として付け足す(`図形テキスト:` 行はキーワード一致に効くので残す)。

- 図形のあるシート(図形 MIN_SHAPES 個以上)を render_drawing.py で PNG にしたもの(シートの末尾に付ける)
- シートに貼られた画像を 1 枚ずつ原寸で取り出したもの(シート全体の PNG は縮小されて文字が読めないため。
  画像だけのシートも対象になる)。画像の上に図形(番号・枠・吹き出し・矢印)が重なっていれば、
  画像と重なる図形の範囲を、画像の元の解像度に近い倍率で描いたもの(元の画像ファイルには図形が写らないため)

出力は Figure(arrows / pairs / facts)の JSON(Structured Outputs)で、行の文面は figure_lines がコードで組み立てる。
矢印の向きは、xlsx / pptx では drawing XML から確定したもの(arrows.py)をプロンプトに渡し、LLM が逆向きに返せば直す。
pdf.py / slides.py も describe()、figure_lines() をここから使う(IMAGE_PROMPT は xlsx の貼り付け画像専用)。

説明文は data/cache/vision/ にキャッシュする(ファイル内容 + 画像のキー + 画像の中身 + モデル + effort + プロンプト
(画像ごとに添えるテキストを含む)が同じなら LLM を呼ばない。描き方を変えたら呼び直す)。API エラーは例外のまま止める(黙って図形テキストだけにはしない)。

モデルは環境変数 VISION_MODEL(既定 gpt-6-luna)、推論の effort は VISION_REASONING_EFFORT
(既定 low。推論モデルでないなら空にする)。
"""

import base64
import hashlib
import io
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openai import OpenAI
from pydantic import BaseModel

from ..vision_config import VISION_MODEL, VISION_REASONING_EFFORT

if TYPE_CHECKING:
    from PIL import Image

from . import Chunked, Section, structured
from .arrows import Arrow as XmlArrow
from .arrows import arrows_from_items, fix_directions, prompt_lines
from .chunking import chunk_sections
from .figure_rules import OMIT, RULES, SCHEMA
from .render_drawing import (
    SCALE,
    Workbook,
    collect,
    open_image,
    render_region,
    render_sheets,
)

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "vision"
MAX_SIDE = 2000  # これより長い辺は縮小して送る
WORKERS = 8  # マルチモーダル LLM の並列数
MIN_PICTURE_PX = (100, 60)  # シート上の表示がこれより小さい画像(アイコン、ロゴ)は読ませない
MAX_PICTURE_SCALE = 4  # 画像を図形ごと描くときの倍率の上限(元の画像の解像度に合わせる)
OVERLAY_MARGIN = 1.0  # 画像の外にはみ出した図形を含めて描く範囲(画像の幅・高さに対する割合)

PROMPT = f"""これは Excel の 1 シートを、セルの文字・図形・貼り付けられた画像ごと画像にしたものです。
図に描かれている内容を、検索で引けるように過不足なく書き出してください。

{RULES}
- 帳票の欄に文字で書かれた値(例:「担当部署 営業部」)も「項目名: 値」の形で facts に書く

{OMIT}

{SCHEMA}
after は使わないので 0 にしてください。"""

IMAGE_PROMPT = f"""これは文書に貼られた画像 1 枚です。画面キャプチャのことが多く、上に番号・枠・吹き出し・矢印が重ねて描かれていることがあります。
検索で引けるように、画像に写っている内容を過不足なく書き出してください。
重ねて描かれた番号・枠・吹き出し・矢印は、それが指している画像の中の箇所との対応を facts に書いてください。

{RULES}
- 表示されているメッセージや説明の文言は、そのまま facts に書き写す
- 画面キャプチャでない図(写真、イラストなど)なら、写っているものと読み取れる数値を facts に書く

{OMIT}

{SCHEMA}
after は使わないので 0 にしてください。"""


# LLM に返させる図の説明の形(Structured Outputs。必ずこの形で返る)。文面は figure_lines が組み立てる
class Arrow(BaseModel):
    source: str  # 元の要素
    target: str  # 先の要素
    label: str  # 線の近くのラベル(無ければ "")
    bidirectional: bool
    after: int  # 本文のどの行の後ろに差し込むか(pdf.py / slides.py のテキスト行の番号。xlsx は使わない)


class Pair(BaseModel):
    upper: str  # 上の箱
    lower: str  # 下の箱(同じ列)
    after: int


class Fact(BaseModel):
    text: str  # 説明 1 行
    after: int


class Figure(BaseModel):
    arrows: list[Arrow]
    pairs: list[Pair]
    facts: list[Fact]

_client = OpenAI()  # 並列で呼ぶので 1 つを共有する


def _describe(png: bytes, prompt: str, schema: type[BaseModel] | None = None) -> str:
    """画像 1 枚を LLM に説明させる。schema を渡せば Structured Outputs でその形の JSON(文字列)を返す。"""
    data_url = "data:image/png;base64," + base64.b64encode(png).decode()
    kwargs: dict[str, Any] = dict(
        model=VISION_MODEL,
        **({"reasoning_effort": VISION_REASONING_EFFORT} if VISION_REASONING_EFFORT else {}),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            }
        ],
    )
    if schema is None:
        res = _client.chat.completions.create(**kwargs)
        return (res.choices[0].message.content or "").strip()
    parsed = _client.chat.completions.parse(response_format=schema, **kwargs).choices[0].message.parsed
    return parsed.model_dump_json() if parsed else "{}"  # parsed が無いのは拒否応答(refusal)。説明なしとして扱う


def png_bytes(im: "Image.Image | bytes") -> bytes:
    """LLM に送る PNG。長い辺が MAX_SIDE を超えていれば縮小する。既に PNG のバイト列ならそのまま返す。"""
    if isinstance(im, bytes):
        return im
    w, h = im.size
    if max(w, h) > MAX_SIDE:
        r = MAX_SIDE / max(w, h)
        im = im.resize((int(w * r), int(h * r)))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def describe(path: Path, images: dict[str, "Image.Image | bytes"], prompt: str,
             extra: dict[str, str] | None = None, schema: type[BaseModel] | None = None) -> dict[str, str]:
    """画像のキー(シート名など)→ 図の説明(LLM の出力)。キャッシュがあればそれを返す。

    images の値は Pillow の画像か、png_bytes で作った PNG のバイト列(PDF はページ数が多く、Pillow の画像のまま全ページ持つと
    1 ページ 8MB 超 × ページ数のメモリを食うので、pdf.py は描いたらすぐ PNG にして渡す)。
    pdf.py / slides.py もこれを使う(プロンプトは呼び出し側が渡す)。
    extra は画像ごとにプロンプトの後ろに添えるテキスト(PDF のページのテキスト行、xlsx / pptx の矢印の一覧など)。
    schema(通常は Figure)を渡せば、その形の JSON で返させる(出力は figure_lines で行にする)。
    キャッシュに無い分は WORKERS 並列で LLM を呼ぶ(PDF は 1 ファイルで数百枚になる)。
    """
    file_hash = hashlib.sha1(path.read_bytes()).hexdigest()
    out, todo = {}, {}
    for name, im in images.items():
        text = prompt + (extra or {}).get(name, "")
        png = png_bytes(im)
        image_hash = hashlib.sha1(png).hexdigest()
        form = schema.__name__ if schema else ""
        key = hashlib.sha1(f"{file_hash}\n{name}\n{image_hash}\n{VISION_MODEL}\n{VISION_REASONING_EFFORT}\n{form}\n{text}".encode()).hexdigest()
        cache = CACHE_DIR / f"{key}.txt"
        if cache.exists():
            out[name] = cache.read_text(encoding="utf-8")
        else:
            todo[name] = (cache, png, text)
    if todo:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(WORKERS) as pool:
            futures = {pool.submit(_describe, png, text, schema): name for name, (_, png, text) in todo.items()}
            for i, future in enumerate(as_completed(futures), 1):
                name = futures[future]
                text = future.result()
                todo[name][0].write_text(text, encoding="utf-8")
                out[name] = text
                print(f"  vision {path.name}: {i}/{len(todo)}", end="\r", file=sys.stderr)
        print(file=sys.stderr)
    return out


def _overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def picture_image(items: list[dict[str, Any]], pic: dict[str, Any]) -> "Image.Image":
    """貼られた画像 1 枚を読ませる画像にする。

    上に図形が重なっていなければ、元の画像ファイルそのもの。重なっていれば、画像の枠と重なる図形の枠を合わせた範囲
    (画像の外へは幅・高さの OVERLAY_MARGIN まで)を render_region で描く。倍率は、元の画像の画素数 / シート上の表示の大きさ
    (元の画像が原寸で写る)にし、SCALE〜MAX_PICTURE_SCALE に収める。EMF / WMF は LibreOffice で PNG にしてから開く
    (render_drawing.open_image)。それでも開けない形式は例外を投げる。
    """
    original = open_image(pic["image"]).convert("RGB")
    over = [it for it in items if it is not pic and it["box"] and it["kind"] != "pic" and _overlaps(pic["box"], it["box"])]
    if not over:
        return original
    x, y, w, h = pic["box"]
    mx, my = w * OVERLAY_MARGIN, h * OVERLAY_MARGIN
    x0 = max(x - mx, min(it["box"][0] for it in over + [pic]))
    y0 = max(y - my, min(it["box"][1] for it in over + [pic]))
    x1 = min(x + w + mx, max(it["box"][0] + it["box"][2] for it in over + [pic]))
    y1 = min(y + h + my, max(it["box"][1] + it["box"][3] for it in over + [pic]))
    scale = min(MAX_PICTURE_SCALE, max(SCALE, original.width / w))
    return render_region(items, (x0, y0, x1 - x0, y1 - y0), scale)


def sheet_pictures(path: Path) -> dict[str, "Image.Image"]:
    """シートに貼られた画像を、`シート名#画像連番` → 読ませる画像(picture_image)で返す(MIN_PICTURE_PX より小さいものは除く)。"""
    wb = Workbook(path)
    pictures = {}
    for sheet in wb.sheets:
        got = collect(wb, sheet)
        if not got:
            continue
        items = got[0]
        pics = [it for it in items if it["kind"] == "pic" and it["image"] and it["box"]]
        pics = [it for it in pics if it["box"][2] >= MIN_PICTURE_PX[0] and it["box"][3] >= MIN_PICTURE_PX[1]]
        for i, it in enumerate(sorted(pics, key=lambda it: (it["box"][1], it["box"][0])), 1):
            try:
                pictures[f"{sheet}#{i}"] = picture_image(items, it)
            except Exception:  # noqa: BLE001  開けない形式(変換もできない EMF など)は飛ばす
                continue
    return pictures


PREFIX = "図の説明: "


def _clean(text: str) -> str:
    return " ".join(text.split())


def figure_lines(text: str, arrows: list[XmlArrow] | None = None) -> list[tuple[int, str]]:
    """Figure の JSON(describe の出力)を (差し込む行の番号, `図の説明:` 行) のリストにする。

    文面はここで決める(LLM には書かせない): 矢印は `A --ラベル--> B`(両向きは両方向)、列の対応は `A の下は B`、
    それ以外は facts の文のまま。さらに、2 つ以上の別の要素から矢印が入る要素ごとに `X に入る矢印の元: A(ラベル) / B(ラベル)`
    をまとめて足す(ある要素への遷移元・入力元を全部聞かれたときに、離れた行を探さずに済むように)。
    arrows(xlsx / pptx の XML から確定した矢印)があれば、LLM が逆向きに返した矢印を先に直す(arrows.fix_directions)。
    壊れた JSON(拒否応答など)は説明なしとして扱う。
    """
    try:
        figure = Figure.model_validate_json(text)
    except ValueError:
        print(f"  vision: JSON を解釈できないので飛ばす: {text[:80]!r}", file=sys.stderr)
        return []
    out: list[tuple[int, str]] = [(f.after, _clean(f.text)) for f in figure.facts if f.text.strip()]
    described = fix_directions(figure.arrows, arrows) if arrows else figure.arrows
    incoming: dict[str, tuple[int, list[str]]] = {}  # 先 → (after, [元(ラベル)])
    for a in described:
        src, dst, label = _clean(a.source), _clean(a.target), _clean(a.label)
        if not src or not dst or "図形なし" in src + dst:  # 端の要素を決められなかった矢印は出さない
            continue
        for s, d in ((src, dst), (dst, src)) if a.bidirectional else ((src, dst),):
            out.append((a.after, f"{s} --{label}--> {d}" if label else f"{s} --> {d}"))
            if s != d:
                incoming.setdefault(d, (a.after, []))[1].append(f"{s}({label})" if label else s)
    for dst, (after, sources) in incoming.items():
        if len(dict.fromkeys(sources)) >= 2:
            out.append((after, f"{dst} に入る矢印の元: " + " / ".join(dict.fromkeys(sources))))
    out += [(p.after, f"{_clean(p.upper)} の下は {_clean(p.lower)}") for p in figure.pairs if p.upper.strip() and p.lower.strip()]
    return [(after, PREFIX + line) for after, line in out if line]


def sheet_arrows(path: Path, sheets: set[str]) -> dict[str, list[XmlArrow]]:
    """シート名 → XML から確定した矢印(arrows.arrows_from_items)。"""
    wb = Workbook(path)
    out = {}
    for sheet in wb.sheets:
        got = collect(wb, sheet) if sheet in sheets else None
        if got:
            out[sheet] = arrows_from_items(got[0])
    return out


def sections(path: Path) -> list[Section]:
    base = structured.sections(path)
    names = {sheet for sheet, _ in base}  # structured が落としたシート(目次、入力規則の選択肢)は読ませない
    images = {k: im for k, im in render_sheets(path).items() if k in names}
    arrows = sheet_arrows(path, set(images))
    extra = {k: prompt_lines(arrows.get(k, [])) for k in images}  # 矢印の向きは XML から確定して渡す
    sheets = describe(path, images, PROMPT, extra=extra, schema=Figure)
    pictures = describe(path, {k: im for k, im in sheet_pictures(path).items() if k.rpartition("#")[0] in names}, IMAGE_PROMPT, schema=Figure)
    sections = []
    for sheet, lines in base:
        if sheet in sheets:
            lines = lines + [(line, "") for _, line in figure_lines(sheets[sheet], arrows.get(sheet))]
        for key in sorted((k for k in pictures if k.rpartition("#")[0] == sheet), key=lambda k: int(k.rpartition("#")[2])):
            lines = lines + [(line, "") for _, line in figure_lines(pictures[key])]
        sections.append((sheet, lines))
    return sections


def extract(path: Path) -> list[Chunked]:
    return chunk_sections(sections(path))
