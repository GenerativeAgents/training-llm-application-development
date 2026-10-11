"""xlsx / pptx の矢印(コネクタ・直線)の向きを、画像ではなく drawing XML から確定する。

マルチモーダル LLM は矢じりの向きを読み違えることがある(ポンチ絵の「申請」の矢印を逆向きに読む、など)。
線の始点・終点と、矢じりがどちらの端に付くか(headEnd / tailEnd)は XML にあるので、それで「元 → 先」を決め、
端点に接続された図形(stCxn / endCxn)、無ければ端点に最も近い「箱」(塗りか枠線のある文字付きの図形)を元と先にする。
矢印のラベル(「入力チェック失敗」など)は塗りも枠線も無い文字だけの図形で、線のそばに置かれるので、箱に限ることで
ラベルを元・先に取り違えない。
確定した矢印はプロンプトに「矢印の一覧」として渡し(LLM はラベルを添えるだけ)、それでも LLM が逆向きに返した
矢印は fix_directions で直す。PDF には図形の情報が無いので使えない(LLM が画像から読む)。

複数の線分をつないだ折れ線(矢じりのある線分の端が、箱ではなく矢じりの無い別の線分につながる)は、
端点が JOIN_PX 以内で重なる線分を箱に届くまでたどる(_follow)。

扱わないもの(矢印の一覧に入らず、LLM が画像から読む):
- ブロック矢印(rightArrow などの図形)
- 端点が箱から離れている線(たどっても NEAR_PX 以内に箱が無い)。その端は「(図形なし)」として渡し、LLM が画像から要素を決める
"""

import math
import re
from typing import Any

from .render_drawing import SCALE, TITLE_PX, Point, connector_path, free_ends

NEAR_PX = 12  # 端点からこの距離(px, 96dpi)以内の図形は、端点がつながっているとみなす
LABEL_PX = 10  # 線からこの距離以内にある文字だけの図形を、その矢印のラベルとみなす
JOIN_PX = 4  # 線分の端点同士がこの距離以内なら、つながった 1 本の折れ線とみなす
MAX_HOPS = 6  # 折れ線をたどる線分の数の上限
CONTAINER_SHAPES = 3  # 文字付きの図形をこれ以上含む図形は大枠(グループの枠)とみなし、矢印の元・先にしない

Arrow = dict[str, Any]  # source / target: 図形の文字(解決できなければ None)、label: 線のそばの文字、bidirectional、p0 / p1: 始点・終点(px)


def _label(it: dict[str, Any]) -> str:
    return " ".join(it["text"].split())


def _center_inside(inner: dict[str, Any], outer: dict[str, Any]) -> bool:
    x, y, w, h = inner["box"]
    ox, oy, ow, oh = outer["box"]
    return ox <= x + w / 2 <= ox + ow and oy <= y + h / 2 <= oy + oh


def _distance(p: Point, box: Any) -> float:
    """点から枠までの距離(枠の中なら 0)。"""
    x, y, w, h = box
    dx = max(x - p[0], 0, p[0] - (x + w))
    dy = max(y - p[1], 0, p[1] - (y + h))
    return math.hypot(dx, dy)


def _resolve(p: Point, connected: str | None, shapes: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """端点 p につながる箱。接続情報があればその図形、無ければ NEAR_PX 以内で一番近い(同じ距離なら小さい)箱。"""
    if connected in by_id and _label(by_id[connected]):
        return by_id[connected]
    near = [(_distance(p, s["box"]), s["box"][2] * s["box"][3], i, s) for i, s in enumerate(shapes)]
    near = [n for n in near if n[0] <= NEAR_PX]
    return min(near)[3] if near else None


def _ends(it: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> tuple[Point, Point] | None:
    """線(コネクタ・直線)の始点と終点。線でなければ None。"""
    if it["kind"] == "cxnSp":
        pts = connector_path(it, by_id)
        return pts[0], pts[-1]
    if it["kind"] == "sp" and it["prst"] == "line":
        return free_ends(it)
    return None


def _follow(p: Point, segments: list[tuple[Point, Point]], shapes: list[dict[str, Any]], by_id: dict[str, dict[str, Any]],
            path: list[Point]) -> dict[str, Any] | None:
    """端点 p から、矢じりの無い線分を端点どうしの重なりでたどり、箱に届いたらその箱。届かなければ None。

    たどった線分の端点は path に足す(ラベルを線の近さで決めるのに使う)。
    """
    used: set[int] = set()
    for _ in range(MAX_HOPS):
        found = _resolve(p, None, shapes, by_id)
        if found is not None:
            return found
        nxt = None
        for i, (a, b) in enumerate(segments):
            if i in used:
                continue
            for near, far in ((a, b), (b, a)):
                if math.dist(p, near) <= JOIN_PX:
                    nxt = (i, far)
                    break
            if nxt:
                break
        if nxt is None:
            return None
        used.add(nxt[0])
        p = nxt[1]
        path.append(p)
    return None


def _segment_distance(box: Any, a: Point, b: Point) -> float:
    """枠から線分 ab までの距離(近似: 線分上の点を細かく取り、枠までの距離の最小)。"""
    n = max(2, int(math.dist(a, b) / 4) + 1)
    return min(_distance((a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n), box) for i in range(n + 1))


def _assign_labels(arrows: list[Arrow], paths: list[list[Point]], labels: list[dict[str, Any]]) -> None:
    """文字だけの図形(ラベル)を、LABEL_PX 以内で一番近い矢印の線に付ける。1 つのラベルは 1 本の矢印にだけ付く。"""
    for lab in labels:
        best = None
        for k, path in enumerate(paths):
            d = min(_segment_distance(lab["box"], a, b) for a, b in zip(path, path[1:]))
            if d <= LABEL_PX and (best is None or d < best[0]):
                best = (d, k)
        if best is not None:
            arrows[best[1]]["label"].append(_label(lab))


def arrows_from_items(items: list[dict[str, Any]]) -> list[Arrow]:
    """render_drawing の図形リストから、矢じりのあるコネクタ・直線を「元 → 先」にして返す(出現順)。"""
    by_id = {it["id"]: it for it in items}
    texts = [it for it in items if it["kind"] in ("sp", "pic", "chart") and it["prst"] != "line" and _label(it)]
    boxes = [s for s in texts if s["fill"] or s["line"]]  # 塗りも枠線も無い文字だけの図形(矢印のラベル)は除く
    shapes = [s for s in boxes if sum(_center_inside(o, s) for o in texts if o is not s) < CONTAINER_SHAPES]
    # 矢じりの無い線分(折れ線の途中の線)。矢じりのある線の端がここにつながっていれば、箱までたどる
    segments = [e for it in items if it["line"] and not (it["head"] or it["tail"]) and (e := _ends(it, by_id)) is not None]
    out: list[Arrow] = []
    paths: list[list[Point]] = []  # 矢印ごとの、たどった線分も含む折れ点列(ラベルの割り当て用)
    for it in items:
        ends = _ends(it, by_id)
        if ends is None or not it["line"] or not (it["head"] or it["tail"]):
            continue  # 線でない、または矢じりの無い線(向きを持たない)
        p0, p1 = ends
        before: list[Point] = []  # p0 側にたどった線分の端点(p0 から遠ざかる順)
        after: list[Point] = []
        src = _resolve(p0, it["stCxn"][0] if it["stCxn"] else None, shapes, by_id) or _follow(p0, segments, shapes, by_id, before)
        dst = _resolve(p1, it["endCxn"][0] if it["endCxn"] else None, shapes, by_id) or _follow(p1, segments, shapes, by_id, after)
        path = before[::-1] + (connector_path(it, by_id) if it["kind"] == "cxnSp" else [p0, p1]) + after
        if it["head"] and not it["tail"]:  # 矢じりが始点側: 終点 → 始点
            src, dst, p0, p1 = dst, src, p1, p0
        out.append({
            "source": _label(src) if src else None,
            "target": _label(dst) if dst else None,
            "label": [],
            "bidirectional": bool(it["head"] and it["tail"]),
            "p0": p0,
            "p1": p1,
        })
        paths.append(path)
    _assign_labels(out, paths, [t for t in texts if t not in boxes])
    for a in out:
        a["label"] = " / ".join(a["label"])
    return out


def pixel(p: Point) -> list[int]:
    """図形の座標(px, 96dpi)を、render_drawing.render が描く画像の画素にする(題の帯の分下げて SCALE 倍)。"""
    return [round(p[0] * SCALE), round((p[1] + TITLE_PX) * SCALE)]


def prompt_lines(arrows: list[Arrow]) -> str:
    """プロンプトの末尾に添える矢印の一覧。無ければ空。"""
    if not arrows:
        return ""
    rows = []
    for n, a in enumerate(arrows, 1):
        src = f"「{a['source']}」" if a["source"] else "(図形なし)"
        dst = f"「{a['target']}」" if a["target"] else "(図形なし)"
        label = f" ラベル「{a['label']}」" if a["label"] else ""
        rows.append(f"A{n} {pixel(a['p0'])}→{pixel(a['p1'])} {src} {'⇔' if a['bidirectional'] else '→'} {dst}{label}")
    return ("\n\n# 矢印の一覧(向きとラベルは図形の情報から確定済み。座標は画像の画素で 始点→終点)\n"
            "この一覧の矢印は、向きとラベルをこのとおりにして arrows に入れてください(ラベルの無いものは画像から読む)。"
            "「(図形なし)」の端は、画像を見てその端に最も近い要素を source / target にしてください(「図形なし」とは書かない)。\n"
            + "\n".join(rows))


def _squash(text: str) -> str:
    return re.sub(r"[\s【】\[\]（）()「」]", "", text)


def _same(a: str, b: str) -> bool:
    a, b = _squash(a), _squash(b)
    return bool(a and b) and (a in b or b in a)


def fix_directions(described: list[Any], arrows: list[Arrow]) -> list[Any]:
    """LLM が返した矢印(source / target / label / bidirectional を持つ pydantic のオブジェクト)を XML の矢印に合わせる。

    逆向きのものは向きを直し、XML でラベルが決まっている矢印は LLM のラベルをそれに置き換える(隣り合う線のラベルを
    取り違えることがある)。要素の名前は LLM が ID を添えるなどして図形の文字と少し違うので、空白・括弧を除いて
    一方が他方を含めば同じとみなす。A → B と B → A の両方が XML にある(往復の矢印)ときは、向きは直さない。
    """
    known = [a for a in arrows if a["source"] and a["target"] and not a["bidirectional"]]
    out = []
    for d in described:
        if not d.bidirectional:
            forward = [a for a in known if _same(d.source, a["source"]) and _same(d.target, a["target"])]
            reverse = [a for a in known if _same(d.source, a["target"]) and _same(d.target, a["source"])]
            if reverse and not forward:
                d = d.model_copy(update={"source": d.target, "target": d.source})
                forward = reverse
            labels = [a["label"] for a in forward if a["label"]]
            if len(forward) == 1 and labels and not _same(d.label or "-", labels[0]):
                d = d.model_copy(update={"label": labels[0]})
        out.append(d)
    return out
