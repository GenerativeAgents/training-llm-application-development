import json
import os
import sys
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import numpy.typing as npt
import weave
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from weave.trace.util import (
    ContextAwareThreadPoolExecutor,  # スレッドの中の op も predict の子として記録する
)

from . import fulltext

load_dotenv(override=True)
PROJECT = os.environ["WANDB_PROJECT"]
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DB_DIR = DATA_DIR / "lancedb"

# 研修では全員が同時に評価を回すので、レートリミット(429)に当たってもリトライして待つ(デフォルトは 2 回)
client = OpenAI(max_retries=10)

SYSTEM_PROMPT = (
    "あなたは、システム開発の設計書、ソフトウェアの操作手順書、行政機関の資料について質問に答えるアシスタントです。"
    "与えられた文書だけを根拠に答えてください。"
    "文書は検索で取り出したチャンクで、質問と関係の無いものも混ざっています。"
    "答えに要る情報が複数の文書に分かれていれば、組み合わせて答えてください。"
    "質問が複数のことを聞いていれば、すべてに答えてください。"
    "文書に手がかりがまったく無いときだけ「分かりません」と答えてください。"
    "一部しか分からないときは、分かる範囲で答え、分からない部分をそう書いてください。"
    "簡潔に答えてください。"
)

# 段階 7: チャンクにある情報を省略・脚色せずに使わせる規則を足す(資料の種類によらない書き方にする)
RULES_PROMPT = SYSTEM_PROMPT.removesuffix("簡潔に答えてください。") + (
    "文言・メッセージ・名称・値そのものを聞かれたら、文書にある表記のまま、省略せずに全部書いてください。"
    "1 つの文言が文書の中で複数の行に分かれて書かれていることがあります。"
    "項目の列挙、手順、処理の流れを聞かれたら、文書にある項目や段階を途中も含めて漏れなく挙げてください。"
    "文書に書かれていない項目や関係(矢印の向き、遷移、順序など)を推測で足さないでください。"
    "前置きや言い換えは省いて簡潔に答えてください。ただし答えの中身は省かないでください。"
)

# RULES_PROMPT に加えて、答える前に根拠の行を抜き書きさせる。判定には answer だけを渡す
EVIDENCE_PROMPT = RULES_PROMPT + (
    "まず、質問に関係する行を文書からそのまま evidence に抜き書きし(関係する行はすべて。途中の段階や続きの行も含める)、"
    "そのあとで抜き書きをもとに回答を answer に書いてください。"
)

# 資料群(collection)ごとに、どういう資料で、チャンクがどう書かれていて、答えをどう書くかを system prompt に足す。
# 資料群の性質から書き、評価の設問からは逆算しない。質問者が資料群を選んだとき(use_collection)だけ使う
COLLECTION_PROMPTS = {
    "システム開発": (
        "この資料群は、業務システム(Web 画面・バッチ・API)の開発で作られた設計書一式です。"
        "要件定義、方式設計(設計標準・テスト標準)、アプリ設計(機能設計書・テーブル定義書・メッセージ設計書・"
        "外部インタフェース設計書・単体テスト仕様書など)の Excel 文書で、表の行は「列名: 値」の形、"
        "シート上の図や貼られた画面キャプチャの内容は「図の説明:」の行で書かれています。"
        "1 つの機能の仕様が複数の設計書に分かれて書かれていることがあります。"
        "単体テスト仕様書は設計書の内容を確かめる文書なので、設計書に書かれていればそちらを根拠にしてください。"
        "画面 ID・機能 ID・テーブルやカラムの物理名・コード値・メッセージは、文書の表記のまま書いてください。"
    ),
    "LibreOffice研修テキスト": (
        "この資料群は、オフィスソフト LibreOffice の操作を画面キャプチャ付きで説明する研修テキスト(PDF)です。"
        "本文は番号付きの手順と画面キャプチャの組で、キャプチャに写っているメニュー・ボタン・入力欄の内容は"
        "「図の説明:」の行で書かれています。手順は複数のページにまたがることがあります。"
        "操作を聞かれたら、最初の操作(メニューを開く、ダイアログを表示する)から順番に書き、"
        "メニュー名・ボタン名・項目名は［ ］の表記のまま書いてください。"
    ),
    "労働保険電子申請マニュアル": (
        "この資料群は、厚生労働省が公開している、e-Gov による労働保険の電子申請の操作マニュアル(PDF)です。"
        "操作は画面キャプチャに吹き出しと引き出し線で示されていて、画面名・欄の名前・ボタン名・表示される値の多くは"
        "キャプチャの中にしかなく、「図の説明:」の行で書かれています。"
        "操作や確認する場所を聞かれたら、どの画面のどの欄・ボタンかを、その画面へのたどり方も含めて書いてください。"
    ),
    "厚労省資料": (
        "この資料群は、厚生労働省が公開している施策・予算事業の概要資料と、統計グラフ中心の普及啓発用スライド集です。"
        "概要資料は箱と矢印の図で事業の実施主体や補助・申請の流れを示していて、図の内容は「図の説明:」の行で書かれています。"
        "グラフの値は、グラフのデータ(「系列: 項目 値」の行)か、画像から読み取った「図の説明:」の行にあります。"
        "金額・率・人数などの数値は資料の値と単位のまま書き、時点や対象(年度、性別、年齢階級など)を添えてください。"
        "流れや関係は、誰から誰へ(矢印の向き)が分かるように書いてください。"
    ),
}

# eval.py --prompt の名前 → RagModel の属性。answer_format が text なら回答をそのまま、evidence なら Structured Outputs の answer を使う
PROMPTS: dict[str, dict[str, Any]] = {
    "base": {"system_prompt": SYSTEM_PROMPT, "answer_format": "text"},
    "rules": {"system_prompt": RULES_PROMPT, "answer_format": "text"},
    "evidence": {"system_prompt": EVIDENCE_PROMPT, "answer_format": "evidence"},
    "collection": {
        "system_prompt": RULES_PROMPT,
        "answer_format": "text",
        "collection_prompts": COLLECTION_PROMPTS,
    },
}


RERANK_PROMPT = (
    "検索で取り出した文書のチャンクが、質問に答えるのにどれだけ役立つかを 0〜10 の整数で score に採点します。"
    "10: 質問の答えそのもの、またはその一部が書かれている。"
    "5: 答えは無いが、答えを探す手がかり(同じ対象についての別の情報、答えのありかを示す記述)がある。"
    "0: 質問と関係が無い。"
    "質問が複数のことを聞いているなら、そのうち 1 つにでも答えているチャンクは高く採点してください。"
)
# 段階 8: チャンクが手順・列挙の途中から始まっているか(そうなら手前のチャンクも渡す)。並べ直しの採点と同じ呼び出しで聞く
MIDWAY_RULE = (
    "チャンクの本文(先頭に付いた見出しの行は除く)が、前の部分から続く手順・操作・列挙の途中から始まっているかを"
    "starts_midway に true / false で答えてください。"
    "たとえば、最初に出てくる手順の番号が 1 でない、前の手順や画面を受けた書き出しになっている、などです。"
)
RERANK_MIDWAY_PROMPT = RERANK_PROMPT + "あわせて、" + MIDWAY_RULE
# 並べ直しの候補に無いチャンク(手前をさかのぼるとき)を判定する
MIDWAY_PROMPT = "文書のチャンクを 1 つ渡します。" + MIDWAY_RULE
# 区切りが互いに独立した文書になっている形式。手前のチャンクへは、区切り(シート)をまたいでさかのぼらない
NO_STEP_BACK_SUFFIXES = (".xlsx",)
EXPAND_COLUMNS = ["id", "title", "text", "collection", "file", "sheet", "unit", "seq"]

RERANK_WORKERS = (
    8  # 1 問の採点を並列に投げる数(Evaluation も問題ごとに並列なので控えめにする)
)


# LLM に返させる JSON の形。Structured Outputs で、必ずこの形で返ってくる
class Score(BaseModel):
    score: int


class ScoreMidway(BaseModel):
    score: int
    starts_midway: bool


class Midway(BaseModel):
    starts_midway: bool


class EvidenceAnswer(BaseModel):
    evidence: list[str]
    answer: str


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quote(value: str) -> str:
    """LanceDB のフィルタ式に埋め込む文字列リテラルを作る。"""
    return "'" + value.replace("'", "''") + "'"


EMBED_BATCH = 32  # 1 リクエストあたりの文書数(トークン上限 30 万に収める)


def embed(texts: list[str], model: str) -> npt.NDArray[np.float64]:
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH):
        res = client.embeddings.create(model=model, input=texts[i : i + EMBED_BATCH])
        embeddings += [d.embedding for d in res.data]
    vectors = np.array(embeddings)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


_tables: dict[
    str, lancedb.table.Table
] = {}  # 開いたテーブルのキャッシュ(index → テーブル)


def table_name(index: str) -> str:
    """インデックス(抽出器の名前)の LanceDB テーブル名。抽出器ごとに別のテーブルにする。"""
    return f"docs_{index}"


def docs_path(index: str) -> Path:
    """ingest.py が書き出す、テーブルに入れる前のチャンク(インデックスの一部として扱い、git 管理しない)。"""
    return DB_DIR / f"{table_name(index)}.jsonl"


def meta_path(index: str) -> Path:
    """build_index.py が最後に書くメタ情報(抽出器、埋め込みモデル)。これがあればインデックスは作り終わっている。"""
    return DB_DIR / f"{table_name(index)}.meta.json"


def has_index(index: str) -> bool:
    return meta_path(index).exists()


def index_meta(index: str) -> dict[str, str]:
    if not has_index(index):
        raise RuntimeError(
            f"インデックス {index} がありません。uv run python -m app.advanced_rag.build_index --extractor {index} を実行してください"
        )
    return json.loads(meta_path(index).read_text(encoding="utf-8"))


def get_table(index: str, embedding_model: str) -> lancedb.table.Table:
    """build_index.py で作った LanceDB テーブルを開く。無ければエラーにする(ここでは作らない)。

    質問側と同じ埋め込みモデルで作られていないと検索結果が無意味になるので、メタ情報と突き合わせる。
    """
    if index not in _tables:
        built_with = index_meta(index)["embedding_model"]
        if built_with != embedding_model:
            raise RuntimeError(
                f"インデックス {index} は {built_with} で作られています({embedding_model} で検索しようとしている)。"
                f"uv run python -m app.advanced_rag.build_index --extractor {index} --embedding-model {embedding_model} で作り直してください"
            )
        _tables[index] = lancedb.connect(DB_DIR).open_table(table_name(index))
    return _tables[index]


class RagModel(weave.Model):
    index: str = "structured"  # 検索するインデックス(作ったときの抽出器の名前。basic / structured / vision)
    chat_model: str = "gpt-6-luna"
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 5
    use_collection: bool = False  # true なら検索対象を collection(data/corpus/ 直下のフォルダ = 資料群)に限定する
    system_prompt: str = SYSTEM_PROMPT
    answer_format: str = "text"  # text(回答をそのまま)/ evidence(Structured Outputs の evidence と answer。system prompt も合わせる)
    collection_prompts: dict[
        str, str
    ] = {}  # collection → system prompt に足す資料群の説明(use_collection のときだけ使う)
    reasoning_effort: str | None = (
        "low"  # 推論モデル(gpt-5 以降)のときだけ指定する。None なら送らない
    )
    rank_depth: int = (
        50  # 診断用に、この順位までの文書 ID を記録する(LLM に渡すのは top_k 件だけ)
    )
    search: str = (
        "vector"  # vector(ベクトル検索)/ fts(全文検索)/ hybrid(両方を RRF で混ぜる)
    )
    fts_fields: list[str] = [
        "morph",
        "bigram",
    ]  # 全文検索で見る列(fulltext.COLUMNS のキー)。複数なら点を足す
    rrf_k: int = 5  # RRF の k。1 / (k + 順位) の和で並べる。小さいほど各検索の上位が強く効く(よく使われる値は 60)
    rerank: str = "none"  # none / llm(検索の上位 rerank_depth 件を LLM が 1 件ずつ採点し、点の順に並べ直す)
    rerank_depth: int = 20  # 並べ直す候補の数。これより下の順位はそのまま後ろに付ける
    select: str = "top_k"  # top_k(上位 top_k 件)/ score(並べ直しの点が score_threshold 以上。rerank="llm" のときだけ)
    score_threshold: int = 7
    min_selected: int = 1  # score で選ぶとき、閾値以上がこれより少なければ上位から補う
    max_selected: int = 5  # score で選ぶときの上限(広げる前の件数)
    expand: str = "none"  # none / unit(同じ区切りのほかのチャンクも渡す)/ unit+steps(さらに、区切りが手順の途中から始まっていれば手前のチャンクも)
    max_back: int = 2  # unit+steps で手前にさかのぼるチャンクの数の上限
    max_context_chars: int = 20000  # 広げるときの、渡すチャンクの文字数の合計の上限(選んだチャンクそのものは必ず渡す)

    @weave.op
    def pos_keywords(self, question: str) -> str:
        """全文検索用のキーワード(空白区切り)を、LLM を使わずに形態素の品詞で抜き出す(fulltext.keywords)。"""
        return fulltext.keywords(question)

    @weave.op
    def rerank_score(self, question: str, title: str, text: str) -> int:
        """チャンク 1 件の、質問に対する関連度(0〜10)。"""
        res = client.chat.completions.parse(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": RERANK_PROMPT},
                {
                    "role": "user",
                    "content": f"# 質問\n{question}\n\n# チャンク: {title}\n{text}",
                },
            ],
            response_format=Score,
            **(
                {"reasoning_effort": self.reasoning_effort}
                if self.reasoning_effort
                else {}
            ),
        )
        return res.choices[0].message.parsed.score

    @weave.op
    def rerank_score_midway(self, question: str, title: str, text: str) -> dict[str, Any]:
        """チャンク 1 件の関連度(score、0〜10)と、手順・列挙の途中から始まっているか(starts_midway)。"""
        res = client.chat.completions.parse(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": RERANK_MIDWAY_PROMPT},
                {
                    "role": "user",
                    "content": f"# 質問\n{question}\n\n# チャンク: {title}\n{text}",
                },
            ],
            response_format=ScoreMidway,
            **(
                {"reasoning_effort": self.reasoning_effort}
                if self.reasoning_effort
                else {}
            ),
        )
        return res.choices[0].message.parsed.model_dump()

    @weave.op
    def starts_midway(self, title: str, text: str) -> bool:
        """チャンクが手順・列挙の途中から始まっているか(並べ直しの候補に無いチャンク用)。"""
        res = client.chat.completions.parse(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": MIDWAY_PROMPT},
                {"role": "user", "content": f"# チャンク: {title}\n{text}"},
            ],
            response_format=Midway,
            **(
                {"reasoning_effort": self.reasoning_effort}
                if self.reasoning_effort
                else {}
            ),
        )
        return res.choices[0].message.parsed.starts_midway

    def _rerank(self, question: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """上位 rerank_depth 件を LLM の点で並べ直す。同点は元の順位の順。score は LLM の点に置き換える。

        expand が unit+steps なら、同じ呼び出しで手順の途中から始まっているか(starts_midway)も付ける。
        """
        head, tail = hits[: self.rerank_depth], hits[self.rerank_depth :]
        if self.expand == "unit+steps":
            judge = lambda h: self.rerank_score_midway(question, h["title"], h["text"])
        else:
            judge = lambda h: {
                "score": self.rerank_score(question, h["title"], h["text"])
            }
        with ContextAwareThreadPoolExecutor(RERANK_WORKERS) as pool:
            got = list(pool.map(judge, head))
        order = sorted(
            range(len(head)), key=lambda i: -got[i]["score"]
        )  # sorted は安定なので同点は元の順
        return [
            head[i] | got[i] | {"score": float(got[i]["score"])} for i in order
        ] + tail

    def _select(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """LLM に渡すチャンクを選ぶ。top_k なら上位 top_k 件、score なら並べ直しの点が閾値以上(min〜max 件)。"""
        if self.select == "top_k":
            return hits[: self.top_k]
        if self.select != "score":
            raise ValueError(f"select は top_k / score のどれか: {self.select!r}")
        if self.rerank != "llm":
            raise ValueError(
                "select=score は rerank=llm のときだけ使える(並べ直しの点で選ぶため)"
            )
        scored = hits[: self.rerank_depth]
        chosen = [h for h in scored if h["score"] >= self.score_threshold][
            : self.max_selected
        ]
        return (
            chosen if len(chosen) >= self.min_selected else scored[: self.min_selected]
        )

    def _expand(
        self, table: lancedb.table.Table, selected: list[dict[str, Any]], judged: dict[str, bool]
    ) -> list[dict[str, Any]]:
        """選んだチャンクを、同じ区切りのほかのチャンク(と、unit+steps なら手前のチャンク)まで広げる。

        選んだチャンクごとに 1 つのまとまり(group)にし、まとまりの中はファイルの中の順(seq)に並べる。
        同じ区切りのチャンクは、選んだチャンクに近い順に max_context_chars まで足す。区切りの先頭のチャンクが手順・列挙の途中から
        始まっていれば(judged は並べ直しで判定済みの分)、ファイルの中で 1 つ前のチャンクを足し、max_back 回までさかのぼる。
        xlsx はシートごとに独立した文書なので、シートをまたいではさかのぼらない(同じシートのチャンクは区切りとして全部足している)。
        """

        def rows(where: str) -> list[dict[str, Any]]:
            return (
                table.search()
                .where(where)
                .select(EXPAND_COLUMNS)
                .limit(10000)
                .to_list()
            )

        included: set[str] = set()
        size = 0
        groups = []
        for hit in selected:
            if hit["id"] in included:
                continue  # 前のまとまりに入っている
            group = [hit | {"role": "selected"}]
            included.add(hit["id"])
            size += len(hit["text"])
            unit = rows(f"file = {quote(hit['file'])} AND unit = {quote(hit['unit'])}")
            for row in sorted(unit, key=lambda r: abs(r["seq"] - hit["seq"])):
                if (
                    row["id"] in included
                    or size + len(row["text"]) > self.max_context_chars
                ):
                    continue
                group.append(row | {"role": "unit"})
                included.add(row["id"])
                size += len(row["text"])
            if self.expand == "unit+steps" and not hit["file"].endswith(
                NO_STEP_BACK_SUFFIXES
            ):
                first = min(unit, key=lambda r: r["seq"])
                current = (
                    first if first["id"] in included else None
                )  # 区切りの先頭を渡していなければさかのぼらない
                for _ in range(self.max_back):
                    if current is None:
                        break
                    midway = judged.get(current["id"])
                    if midway is None:
                        midway = self.starts_midway(current["title"], current["text"])
                    if not midway:
                        break
                    before = rows(
                        f"file = {quote(current['file'])} AND seq = {current['seq'] - 1}"
                    )
                    if (
                        not before
                        or before[0]["id"] in included
                        or size + len(before[0]["text"]) > self.max_context_chars
                    ):
                        break
                    current = before[0]
                    group.append(current | {"role": "before"})
                    included.add(current["id"])
                    size += len(current["text"])
            groups.append(sorted(group, key=lambda r: r["seq"]))
        return [c | {"group": g} for g, group in enumerate(groups) for c in group]

    def _vector_hits(
        self, table: lancedb.table.Table, question: str, where: str | None, depth: int
    ) -> list[dict[str, Any]]:
        query = embed([question], self.embedding_model)[0]
        search = (
            table.search(query, vector_column_name="vector")
            .distance_type("cosine")
            .limit(depth)
        )
        if where:
            search = search.where(where, prefilter=True)
        # score はコサイン類似度(1 - cosine 距離)
        return [h | {"score": 1.0 - float(h["_distance"])} for h in search.to_list()]

    def _fts_hits(
        self, table: lancedb.table.Table, question: str, where: str | None, depth: int
    ) -> list[dict[str, Any]]:
        query = fulltext.query(self.pos_keywords(question), self.fts_fields)
        if query is None:
            return []
        search = table.search(query, query_type="fts").limit(depth)
        if where:
            search = search.where(where, prefilter=True)
        return [
            h | {"score": float(h["_score"])} for h in search.to_list()
        ]  # score は BM25

    @weave.op
    def retrieve(self, question: str, collection: str | None = None) -> dict[str, Any]:
        """上位 top_k 件のチャンク(contexts)と、上位 rank_depth 件の ID(ranked_ids、正解の順位の診断用)を返す。"""
        table = get_table(self.index, self.embedding_model)
        depth = max(self.top_k, self.rank_depth)
        # 文書の collection 列(メタデータ)で先に絞ってから、その中で検索する(prefilter)
        where = (
            f"collection = {quote(collection)}"
            if self.use_collection and collection
            else None
        )
        if self.search == "vector":
            hits = self._vector_hits(table, question, where, depth)
        elif self.search == "fts":
            hits = self._fts_hits(table, question, where, depth)
        elif self.search == "hybrid":
            # 各検索の上位 depth 件を、順位だけで混ぜる(RRF)。score は RRF の値
            lists = [
                self._vector_hits(table, question, where, depth),
                self._fts_hits(table, question, where, depth),
            ]
            docs: dict[str, dict[str, Any]] = {}
            scores: dict[str, float] = {}
            for hits_ in lists:
                for rank, h in enumerate(hits_, 1):
                    docs.setdefault(h["id"], h)
                    scores[h["id"]] = scores.get(h["id"], 0.0) + 1.0 / (
                        self.rrf_k + rank
                    )
            hits = [
                docs[i] | {"score": scores[i]}
                for i in sorted(scores, key=lambda i: -scores[i])
            ][:depth]
        else:
            raise ValueError(
                f"search は vector / fts / hybrid のどれか: {self.search!r}"
            )
        if self.rerank == "llm":
            hits = self._rerank(question, hits)
        elif self.rerank != "none":
            raise ValueError(f"rerank は none / llm のどれか: {self.rerank!r}")
        selected = self._select(hits)
        if self.expand in ("unit", "unit+steps"):
            judged = {h["id"]: h["starts_midway"] for h in hits if "starts_midway" in h}
            selected = self._expand(table, selected, judged)
        elif self.expand != "none":
            raise ValueError(
                f"expand は none / unit / unit+steps のどれか: {self.expand!r}"
            )
        drop = {
            "vector",
            "_distance",
            "_score",
            "_relevance_score",
            *fulltext.ALL_COLUMNS,
        }
        contexts = [{k: v for k, v in h.items() if k not in drop} for h in selected]
        return {"contexts": contexts, "ranked_ids": [h["id"] for h in hits]}

    @weave.op
    def predict(self, question: str, collection: str | None = None) -> dict[str, Any]:
        """collection は質問者が選んだ資料群(データセットの collection 列)。use_collection が false なら使わない。"""
        retrieved = self.retrieve(question, collection)
        contexts = retrieved["contexts"]
        # 広げたとき(group あり)は、まとまりごとに 1 つの文書として渡す。題は区切り(unit)の並び
        blocks: dict[int, list[dict[str, Any]]] = {}
        for n, c in enumerate(contexts):
            blocks.setdefault(c.get("group", n), []).append(c)
        titles = {
            g: cs[0]["title"]
            if len(cs) == 1
            else cs[0]["title"].rpartition(" / ")[0]
            + " / "
            + ", ".join(dict.fromkeys(c["unit"] for c in cs))
            for g, cs in blocks.items()
        }
        context_text = "\n\n".join(
            f"## {titles[g]}\n" + "\n".join(c["text"] for c in cs)
            for g, cs in blocks.items()
        )
        system_prompt = self.system_prompt
        if self.use_collection and collection in self.collection_prompts:
            system_prompt += (
                "\n\n# 資料群について\n" + self.collection_prompts[collection]
            )
        request = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"# 文書\n{context_text}\n\n# 質問\n{question}",
                },
            ],
            **(
                {"reasoning_effort": self.reasoning_effort}
                if self.reasoning_effort
                else {}
            ),
        }
        output = {"contexts": contexts, "ranked_ids": retrieved["ranked_ids"]}
        if self.answer_format == "text":
            res = client.chat.completions.create(**request)
            return {"answer": res.choices[0].message.content, **output}
        if self.answer_format == "evidence":
            res = client.chat.completions.parse(
                **request, response_format=EvidenceAnswer
            )
            parsed = res.choices[0].message.parsed
            return {"answer": parsed.answer, **output, "evidence": parsed.evidence}
        raise ValueError(
            f"answer_format は text / evidence のどれか: {self.answer_format!r}"
        )


if __name__ == "__main__":
    weave.init(os.environ["WANDB_PROJECT"])
    question = sys.argv[1] if len(sys.argv) > 1 else "顧客検索APIのHTTPメソッドは?"
    output = RagModel().predict(question)
    print("Q:", question)
    print("A:", output["answer"])
    for c in output["contexts"]:
        print(f"  - {c['id']} ({c['score']:.3f})")
