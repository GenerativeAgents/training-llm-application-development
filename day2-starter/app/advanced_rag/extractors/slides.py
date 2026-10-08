"""pptx の抽出器。スライド 1 枚を 1 区切り(xlsx のシート相当)にし、ID は `相対パス#slide08`。

図解の多いスライド(省庁のポンチ絵など)は本文がほぼ図形の中にあり、1 枚に箱・矢印・表が数十個並ぶ。文字はどの抽出器でも全部取れるが、
「どの箱の文字か」「箱同士の関係(矢印)」がどこまで残るかが抽出器で違う:

- basic(basic.py): Microsoft markitdown の素の出力。図形の XML 出現順(= 重なり順)なので、右上の注記が題より先に出るなど読み順が崩れる
- structured(extract): render_drawing.Collector で図形を位置つきで集め、上から下・左から右に並べ直す。
  表は行ごとに ` | ` で連結。グラフは題と系列ごとの値の行(`系列名: 項目 値 / …`)にする。矢印の向きは落ちる
- vision: 上に加えて、スライドを render_drawing.py で PNG にし、structured の行(図形の枠の座標付き)と一緒にマルチモーダル LLM に渡す。
  図形の文字は書き写させず、矢印・入れ子・配置で表された関係だけを 1 事実 1 行の JSON で返させ、
  指定された行の後ろに `図の説明:` 行として差し込む(pdf.py の vision と同じやり方)
"""

import re
from pathlib import Path
from typing import Any

from . import Chunked, Line, Section
from .chunking import chunk_sections
from .figure_rules import OMIT, RULES
from .render_drawing import (
    SCALE,
    TITLE_PX,
    Presentation,
    chart_lines,
    collect_slide,
    render_slides,
)

ROW_PX = 24  # この高さ(px, 96dpi)で行の帯を作り、帯の中は左から右へ並べる

TABLE_CELL_RE = re.compile(r"^(\d+)-(\d+)-(\d+)$")  # render_drawing.Collector.emit_table のセル ID: 表ID-行-列


def text_top(it: dict[str, Any]) -> float:
    """文字が始まる高さ(px)。枠の上端ではなくこれで並べる。

    大枠(上端が高い)の中に小さな見出しラベルが重なっている構図が多く、枠の上端で並べると中身が見出しより先に来る。
    縦位置が中央/下の図形は、段落数と文字サイズから文字の高さを見積もって位置を出す。
    """
    x, y, w, h = it["box"]
    th = sum(pt * 96 / 72 * 1.2 for _, pt, _ in it["paras"]) or 0
    if it["valign"] == "middle":
        return y + max(0, (h - th) / 2)
    if it["valign"] == "bottom":
        return y + max(0, h - th)
    return y + it["ins"][1]


Box = tuple[float, float, float, float]  # (x0, top, x1, bottom)(px, 96dpi)


def boxed_lines(items: list[dict[str, Any]]) -> list[tuple[Box, Line]]:
    """図形の文字を読み順(上から下、同じ帯なら左から右)に行にし、行ごとにその図形の枠を添える。

    段落ごとに 1 行(同じ図形の段落は同じ枠)。表はセルを行ごとに ` | ` で連結して 1 行にし、1 行目をヘッダ扱いにする
    (枠はその行のセルを合わせた範囲)。最初の行を題(heading)にする(ポンチ絵の題は最上段にある)。
    """
    units: list[tuple[tuple[float, float], Box, list[str], str]] = []  # ((文字の上端, 左端), 枠, 行のリスト, 種別)
    rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for it in items:
        m = TABLE_CELL_RE.match(it["id"])
        if m:
            rows.setdefault((m.group(1), int(m.group(2))), []).append(it)
        elif it["kind"] == "chart":  # グラフは題・系列ごとの値の行(render_drawing.chart_lines)。枠の上端で並べる
            x, y, w, h = it["box"]
            units.append(((y, x), (x, y, x + w, y + h), chart_lines(it["chart"]), ""))
        elif it["text"].strip():
            x, y, w, h = it["box"]
            units.append(((text_top(it), x), (x, y, x + w, y + h), [" ".join(p.split()) for p in it["text"].split("\n")], ""))
    for (_, r), cells in rows.items():
        cells.sort(key=lambda c: c["box"][0])
        texts = [" ".join(c["text"].split()) for c in cells]
        pos = (min(c["box"][1] for c in cells), cells[0]["box"][0])
        box = (min(c["box"][0] for c in cells), min(c["box"][1] for c in cells),
               max(c["box"][0] + c["box"][2] for c in cells), max(c["box"][1] + c["box"][3] for c in cells))
        units.append((pos, box, [" | ".join(t for t in texts if t)], "header" if r == 0 else ""))
    units.sort(key=lambda u: (int(u[0][0] // ROW_PX), u[0][1]))
    lines: list[tuple[Box, Line]] = []
    for _, box, paras, kind in units:
        for p in paras:
            if p:
                lines.append((box, (p, kind or ("heading" if not lines else ""))))
    return lines


def slide_lines(items: list[dict[str, Any]]) -> list[Line]:
    return [line for _, line in boxed_lines(items)]


def sections(path: Path) -> list[Section]:
    pres = Presentation(path)
    return [(name, slide_lines(collect_slide(pres, part))) for name, part in pres.slides]


# --- vision -------------------------------------------------------------------------------------

PROMPT = f"""これはプレゼンテーション資料の 1 スライドを画像にしたものと、スライドの図形から取り出した文字の行です。
テキスト行は `L番号 [x0,top,x1,bottom] 文字` の形で、座標はその文字が入っている図形(枠・箱・表の行)の範囲を画像の画素で表したものです。
1 つの図形に段落が複数あれば、同じ座標の行が続きます。
図形の文字は別に取り出してあるので、書き写さないでください。図と画像が表している内容を、検索で引けるように説明してください。

文字の読み方:
- 画像に見えている文字は、同じ位置にあるテキスト行の文字を正とする(画像では潰れて読みにくい文字も、テキスト行で確かめる)
- 貼り付けられた画像の中の文字はテキスト行に無いので、画像から読む

書くこと:
{RULES}

{OMIT}

JSON で {{"items": [{{"after": 行番号, "text": "説明 1 行"}}, ...]}} の形で返してください。
1 つの item には 1 つの事実だけを書き、段落にまとめないでください(矢印 1 本、所属 1 つ、対応 1 つ、画面の項目 1 つがそれぞれ 1 item)。
after は、その説明を本文のどの行の後ろに差し込むかで、説明している図の見出し(枠の題など)の行の番号(L の後ろの数字)にします。
図の構造や画像で表されている情報が無ければ {{"items": []}} を返してください。"""


def extract(path: Path) -> list[Chunked]:
    return chunk_sections(sections(path))


def sections_vision(path: Path) -> list[Section]:
    """スライドの画像と、図形の文字の行(画像上の座標付き)を一緒にマルチモーダル LLM に渡し、説明を指定の行の後ろに差し込む。

    テキスト行は structured の行(slide_lines)と同じ行・同じ順なので、LLM が返す行番号がそのまま本文の行を指す。
    行番号が範囲外なら、スライドの末尾に付ける。
    """
    from . import vision  # openai が要るので、使うときだけ読み込む

    pres = Presentation(path)
    images = render_slides(path)
    boxed = {name: boxed_lines(collect_slide(pres, part)) for name, part in pres.slides}

    def pixel(box: Box) -> list[int]:  # render_drawing と同じく、題の帯の分下げて SCALE 倍する
        x0, top, x1, bottom = box
        return [round(v * SCALE) for v in (x0, top + TITLE_PX, x1, bottom + TITLE_PX)]

    extra = {name: "\n\n# テキスト行\n" + "\n".join(f"L{n} {pixel(box)} {text}" for n, (box, (text, _)) in enumerate(lines, 1))
             for name, lines in boxed.items() if name in images}
    described = vision.describe(path, images, PROMPT, extra=extra, json_mode=True)
    result = []
    for name, lines in boxed.items():
        groups: dict[int, list[str]] = {}
        for after, text in vision.json_items(described.get(name, "")) if name in described else []:
            groups.setdefault(after if 1 <= after <= len(lines) else len(lines), []).append(text)
        out: list[Line] = []
        for n, (_, line) in enumerate(lines, 1):
            out.append(line)
            out += vision.description_lines("\n".join(groups.get(n, [])))
        result.append((name, out))
    return result


def extract_vision(path: Path) -> list[Chunked]:
    return chunk_sections(sections_vision(path))
