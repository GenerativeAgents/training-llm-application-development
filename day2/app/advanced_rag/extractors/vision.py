"""structured 抽出器に、図や画像の「図の説明」をマルチモーダル LLM で足す抽出器(段階 4)。

structured は図形の文字だけを出現順に並べるので、矢印の向きやチェックボックスの状態が失われる。
貼り付けられた画像(画面キャプチャ)の中の文字も取れない。ここでは次の 2 つをマルチモーダル LLM に説明させ、
`図の説明:` 行として付け足す(`図形テキスト:` 行はキーワード一致に効くので残す)。

- 図形のあるシート(図形 MIN_SHAPES 個以上)を render_drawing.py で PNG にしたもの(シートの末尾に付ける)
- シートに貼られた画像を 1 枚ずつ原寸で取り出したもの(シート全体の PNG は縮小されて文字が読めないため。
  画像だけのシートも対象になる)。画像の上に図形(番号・枠・吹き出し・矢印)が重なっていれば、
  画像と重なる図形の範囲を、画像の元の解像度に近い倍率で描いたもの(元の画像ファイルには図形が写らないため)

pdf.py / slides.py も describe() と json_items() をここから使う(IMAGE_PROMPT は xlsx の貼り付け画像専用)。

説明文は data/cache/vision/ にキャッシュする(ファイル内容 + 画像のキー + 画像の中身 + モデル + effort + プロンプト
(画像ごとに添えるテキストを含む)が同じなら LLM を呼ばない。描き方を変えたら呼び直す)。API エラーは例外のまま止める(黙って図形テキストだけにはしない)。

モデルは環境変数 VISION_MODEL(既定 gpt-6-luna)、推論の effort は VISION_REASONING_EFFORT
(既定 low。推論モデルでないなら空にする)。
"""

import base64
import hashlib
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from . import Chunked, Section
from . import structured
from .chunking import chunk_sections
from .figure_rules import OMIT, RULES

from .render_drawing import SCALE, Workbook, collect, open_image, render_region, render_sheets

VISION_MODEL = os.environ.get("VISION_MODEL", "gpt-6-luna")
VISION_REASONING_EFFORT = os.environ.get("VISION_REASONING_EFFORT", "low")
CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "vision"
MAX_SIDE = 2000  # これより長い辺は縮小して送る
WORKERS = 8  # マルチモーダル LLM の並列数
MIN_PICTURE_PX = (100, 60)  # シート上の表示がこれより小さい画像(アイコン、ロゴ)は読ませない
MAX_PICTURE_SCALE = 4  # 画像を図形ごと描くときの倍率の上限(元の画像の解像度に合わせる)
OVERLAY_MARGIN = 1.0  # 画像の外にはみ出した図形を含めて描く範囲(画像の幅・高さに対する割合)

PROMPT = f"""これは Excel の 1 シートを、セルの文字・図形・貼り付けられた画像ごと画像にしたものです。
図に描かれている内容を、検索で引けるように過不足なく箇条書きにしてください。

{RULES}
- 帳票の欄に文字で書かれた値(例:「担当部署 営業部」)も「項目名: 値」の形で書く

{OMIT}
前置きや見出しは不要で、箇条書きの行だけを出力してください。"""

IMAGE_PROMPT = f"""これは文書に貼られた画像 1 枚です。画面キャプチャのことが多く、上に番号・枠・吹き出し・矢印が重ねて描かれていることがあります。
検索で引けるように、画像に写っている内容を過不足なく箇条書きにしてください。
重ねて描かれた番号・枠・吹き出し・矢印は、それが指している画像の中の箇所との対応を書いてください。

{RULES}
- 表示されているメッセージや説明の文言は、そのまま書き写す
- 画面キャプチャでない図(写真、イラストなど)なら、写っているものと読み取れる数値を書く

{OMIT}
前置きや見出しは不要で、箇条書きの行だけを出力してください。"""

_client = OpenAI()  # 並列で呼ぶので 1 つを共有する


def _describe(png: bytes, prompt: str, json_mode: bool = False) -> str:
    data_url = "data:image/png;base64," + base64.b64encode(png).decode()
    res = _client.chat.completions.create(
        model=VISION_MODEL,
        **({"reasoning_effort": VISION_REASONING_EFFORT} if VISION_REASONING_EFFORT else {}),
        **({"response_format": {"type": "json_object"}} if json_mode else {}),
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
    return (res.choices[0].message.content or "").strip()


def _png_bytes(im) -> bytes:
    w, h = im.size
    if max(w, h) > MAX_SIDE:
        r = MAX_SIDE / max(w, h)
        im = im.resize((int(w * r), int(h * r)))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def describe(path: Path, images: dict[str, "Image.Image"], prompt: str,
             extra: dict[str, str] | None = None, json_mode: bool = False) -> dict[str, str]:
    """画像のキー(シート名など)→ 図の説明(LLM の出力)。キャッシュがあればそれを返す。

    pdf.py / slides.py もこれを使う(プロンプトは呼び出し側が渡す)。
    extra は画像ごとにプロンプトの後ろに添えるテキスト(PDF のページのテキスト行など)。
    json_mode なら JSON で返させる(出力は json_items で解釈する)。
    キャッシュに無い分は WORKERS 並列で LLM を呼ぶ(PDF は 1 ファイルで数百枚になる)。
    """
    file_hash = hashlib.sha1(path.read_bytes()).hexdigest()
    out, todo = {}, {}
    for name, im in images.items():
        text = prompt + (extra or {}).get(name, "")
        png = _png_bytes(im)
        image_hash = hashlib.sha1(png).hexdigest()
        key = hashlib.sha1(f"{file_hash}\n{name}\n{image_hash}\n{VISION_MODEL}\n{VISION_REASONING_EFFORT}\n{text}".encode()).hexdigest()
        cache = CACHE_DIR / f"{key}.txt"
        if cache.exists():
            out[name] = cache.read_text()
        else:
            todo[name] = (cache, png, text)
    if todo:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(WORKERS) as pool:
            futures = {pool.submit(_describe, png, text, json_mode): name for name, (_, png, text) in todo.items()}
            for i, future in enumerate(as_completed(futures), 1):
                name = futures[future]
                text = future.result()
                todo[name][0].write_text(text)
                out[name] = text
                print(f"  vision {path.name}: {i}/{len(todo)}", end="\r", file=sys.stderr)
        print(file=sys.stderr)
    return out


def _overlaps(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def picture_image(items: list[dict], pic: dict) -> "Image.Image":
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


def description_lines(text: str) -> list[tuple[str, str]]:
    """LLM の箇条書きを `図の説明:` 行にする。"""
    return [("図の説明: " + line.lstrip("-*・ ").strip(), "") for line in text.splitlines() if line.strip()]


def json_items(text: str) -> list[tuple[int, str]]:
    """JSON で返させた説明({"items": [{"after": 行番号, "text": ...}]})を (差し込む行の番号, 説明 1 行) のリストにする。

    pdf.py / slides.py が使う。行番号が無い・数でなければ 0(呼び出し側で末尾扱い)。壊れた JSON は説明なしとして扱う。
    """
    try:
        items = json.loads(text).get("items", [])
    except (json.JSONDecodeError, AttributeError):
        print(f"  vision: JSON を解釈できないので飛ばす: {text[:80]!r}", file=sys.stderr)
        return []
    result = []
    for item in items:
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            continue
        try:
            after = int(item.get("after") or 0)
        except (TypeError, ValueError):
            after = 0
        result.append((after, " ".join(str(item["text"]).split())))
    return result


def sections(path: Path) -> list[Section]:
    base = structured.sections(path)
    names = {sheet for sheet, _ in base}  # structured が落としたシート(目次、入力規則の選択肢)は読ませない
    sheets = describe(path, {k: im for k, im in render_sheets(path).items() if k in names}, PROMPT)
    pictures = describe(path, {k: im for k, im in sheet_pictures(path).items() if k.rpartition("#")[0] in names}, IMAGE_PROMPT)
    sections = []
    for sheet, lines in base:
        if sheet in sheets:
            lines = lines + description_lines(sheets[sheet])
        for key in sorted((k for k in pictures if k.rpartition("#")[0] == sheet), key=lambda k: int(k.rpartition("#")[2])):
            lines = lines + description_lines(pictures[key])
        sections.append((sheet, lines))
    return sections


def extract(path: Path) -> list[Chunked]:
    return chunk_sections(sections(path))
