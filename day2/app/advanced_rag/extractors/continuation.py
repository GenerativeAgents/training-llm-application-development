"""隣り合う位置単位(ページ / スライド)が 1 つの手順・列挙・表の続きかを判定し、続く限りまとめて 1 つの区切りにする。

しおりの無い(または 1 階層だけの)PDF と pptx は、ページ / スライドがそのまま区切りになる。操作マニュアルの手順は複数ページに
またがるので、そのままだと手順の途中のページが単独の区切りになり、手順名・画面名の見出しが付かず検索に当たりにくく、
回答時(段階 8 の拡張)にも手順の一部しか渡らない。

判定は、決定的な手がかりがあればそれで決め、無いときだけ LLM に任せる(arrows.py が矢印の向きを XML で確定して LLM より優先するのと
同じ考え方):
- B の先頭が題らしい行(手順番号が無い短い行で、文で終わっていない。画面名・章題)なら、B は新しい区切り
- B の先頭が手順番号 1 なら新しい手順。A の最後の手順番号 + 1 なら A の続き
- どちらでも決まらなければ(番号の無い表・箇条書きの続きなど)、2 つの本文を渡して「B は A の続きか」を Decisions API の predicate で問い、
  確率が THRESHOLD 以上なら続きとする。文書の構造の判定なので質問文は渡さない(質問を混ぜると、質問に関係のあるページが
  「続き」に見えて判定が歪む)
手順番号が文書全体で通しのマニュアルでは、番号だけ見ると画面をまたいでも全部続きになるので、題の行の規則を先に見る。
しおりが 2 階層以上ある PDF はしおりの項目を区切りにするので、ここは使わない。
LLM の判定は data/cache/continuation/ にキャッシュする(A・B の本文、モデル、問の文面がキー)。閾値や上限を変えても再呼び出しは要らない。
"""

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "continuation"
MODEL = "gpt-6-luna"  # Decisions API が対応するモデル(これだけ)
THRESHOLD = 0.5  # 「続き」とみなす確率の下限
MAX_GROUP = 10  # 1 区切りにまとめる位置単位の数の上限(区切りが際限なく伸びないように)
WORKERS = 8  # 判定の並列数
QUESTION = (
    "チャンク B の本文(先頭に付いた見出しの行は除く)は、チャンク A の本文の続きである。"
    "つまり、A で始まった、または A が途中にある 1 つの手順・操作の流れ・列挙・表が、B でそのまま続いている。"
    "B で別の手順・別の話題・別の図表が始まっているなら false。"
)
STEP_RE = re.compile(r"^[（(]?([0-9０-９]{1,3})[）)．.、]\s*\S")  # "３９．以下の手順" / "(3) …" / "3. …"
TITLE_MAX_CHARS = 30  # 題らしい行の長さの上限
SENTENCE_ENDS = ("。", "．", "ください", "ます", "です")  # これで終わる行は文(題ではない)

_client: OpenAI | None = None  # 使うときに作る(読み込むだけで OPENAI_API_KEY を要求しない)


def step_no(line: str) -> int | None:
    """行の先頭の手順番号(無ければ None)。全角の数字も読む。"""
    m = STEP_RE.match(line.strip())
    return int(m[1].translate(str.maketrans("０１２３４５６７８９", "0123456789"))) if m else None


def is_title(line: str) -> bool:
    """題らしい行か: 手順番号が無い短い行で、文で終わっていない(画面名、章題、表の題)。"""
    line = line.strip()
    return bool(line) and len(line) <= TITLE_MAX_CHARS and step_no(line) is None and not line.endswith(SENTENCE_ENDS)


def by_rule(a: str, b: str) -> bool | None:
    """決定的な手がかりで B が A の続きかが決まるなら True / False、決まらなければ None。"""
    b_lines = [line for line in b.splitlines() if line.strip()]
    if not b_lines:
        return None
    if is_title(b_lines[0]):
        return False
    first = step_no(b_lines[0])
    if first is None:
        return None
    if first == 1:
        return False
    last = next((n for n in (step_no(line) for line in reversed(a.splitlines())) if n is not None), None)
    return True if last is not None and first == last + 1 else None


def _decide(a: str, b: str) -> float:
    """B が A の続きである確率(LLM)。答えられなかったとき(refusal)は 0。"""
    global _client
    if _client is None:
        _client = OpenAI()
    res = _client.decisions.create(
        model=MODEL,
        input=f"# チャンク A\n{a}\n\n# チャンク B\n{b}",
        questions=[{"type": "predicate", "name": "continues", "instructions": QUESTION}],
    )
    answer = res.answers[0]
    return answer.probability if answer.type == "predicate" else 0.0


def continuation_probs(pairs: list[tuple[str, str]]) -> list[float]:
    """組 (A, B) ごとの、B が A の続きである確率(LLM)。キャッシュに無い組だけ判定する。"""
    probs: list[float | None] = [None] * len(pairs)
    todo: dict[int, Path] = {}
    for i, (a, b) in enumerate(pairs):
        key = hashlib.sha1(f"{MODEL}\n{QUESTION}\n{a}\n\x00\n{b}".encode()).hexdigest()
        cache = CACHE_DIR / f"{key}.txt"
        if cache.exists():
            probs[i] = float(cache.read_text(encoding="utf-8"))
        else:
            todo[i] = cache
    if todo:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(WORKERS) as pool:
            got = pool.map(lambda i: _decide(*pairs[i]), todo)
        for i, p in zip(todo, got):
            todo[i].write_text(f"{p:.4f}", encoding="utf-8")
            probs[i] = p
    return [p if p is not None else 0.0 for p in probs]


def group_indices(texts: list[str]) -> list[list[int]]:
    """位置単位の本文の列を、続きどうしでまとめた添字の組に分ける(順番は保つ)。1 組は MAX_GROUP 個まで。"""
    if not texts:
        return []
    decisions = [by_rule(a, b) for a, b in zip(texts, texts[1:])]
    undecided = [i for i, d in enumerate(decisions) if d is None]
    for i, p in zip(undecided, continuation_probs([(texts[i], texts[i + 1]) for i in undecided])):
        decisions[i] = p >= THRESHOLD
    groups = [[0]]
    for i, continues in enumerate(decisions, 1):
        if continues and len(groups[-1]) < MAX_GROUP:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups
