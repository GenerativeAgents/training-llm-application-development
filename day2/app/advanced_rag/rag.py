"""Advanced RAG の検索と回答生成(RagModel)。インデックスは build_index.py で先に作っておく。

predict() は次の順に処理する。どの処理を使うかは RagModel のフィールドで切り替え、研修の段階
(pages/part3_1_advanced_rag.py の STAGES)はその組み合わせで表す:

1. 検索(search): ベクトル検索(vector)か、ベクトル検索と全文検索を RRF で混ぜる(hybrid、段階 5〜)。
   use_collection なら、質問者が選んだ資料群(collection)に先に絞る(段階 2〜)
2. 並べ直し(rerank): 上位のチャンクを Decisions API の 1 回の呼び出しで採点し、点の順に並べ直す(llm、段階 6〜)
3. 選択(select): LLM に渡すチャンクを、上位 top_k 件か、並べ直しの点が閾値以上のもので選ぶ(score、段階 8)
4. 拡張(expand): 選んだチャンクに、同じ区切り(unit)のほかのチャンクを足す(unit、段階 8)。ページをまたぐ手順は索引時に
   1 つの区切りにまとめてある(extractors/continuation.py)ので、検索時は区切りをそのまま使う
5. 回答: system prompt(段階 7 で IMPROVED_PROMPT と資料群ごとの説明に置き換える)とチャンクを渡して答えさせる

retrieve() は 1〜4 だけを行う。predict() は答えと一緒に、渡したチャンク(contexts)と検索の順位(ranked_ids)も返し、
評価(notebooks/part3_1_eval.ipynb)はそれで検索の良し悪しも測る。

    uv run python -m app.advanced_rag.rag "プロジェクト登録画面の入力項目は?"
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import lancedb
import numpy as np
import numpy.typing as npt
import weave
from dotenv import load_dotenv
from lancedb.query import LanceVectorQueryBuilder
from openai import OpenAI
from openai.types.decision import Decision
from openai.types.decision_create_params import QuestionQuestionParamScoreLevel
from openai.types.shared import ReasoningEffort

from . import fulltext

load_dotenv(override=True)
PROJECT = os.environ["WANDB_PROJECT"]
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DB_DIR = DATA_DIR / "lancedb"

# 研修では全員が同時に評価を回すので、レートリミット(429)に当たってもリトライして待つ(デフォルトは 2 回)
client = OpenAI(max_retries=10)

# 段階 1〜6 の回答用の system prompt(RagModel の既定)
BASIC_PROMPT = (
    "あなたは、システム開発の設計書、ソフトウェアの操作手順書、行政機関の資料について質問に答えるアシスタントです。"
    "与えられた文書だけを根拠に答えてください。"
    "文書は検索で取り出したチャンクで、質問と関係の無いものも混ざっています。"
    "答えに要る情報が複数の文書に分かれていれば、組み合わせて答えてください。"
    "質問が複数のことを聞いていれば、すべてに答えてください。"
    "文書に手がかりがまったく無いときだけ「分かりません」と答えてください。"
    "一部しか分からないときは、分かる範囲で答え、分からない部分をそう書いてください。"
    "簡潔に答えてください。"
)

# 段階 7 で BASIC_PROMPT と置き換える。チャンクにある情報を省略・脚色せずに使わせる規則を入れる(資料の種類によらない書き方にする)。
# どういう資料かは資料群ごとの説明(COLLECTION_PROMPTS)に任せるので、役割の文から資料の種類を外す
IMPROVED_PROMPT = (
    "あなたは、検索で取り出した文書をもとに質問に答えるアシスタントです。"
    "文書には質問と関係の無いものも混ざっています。"
    "文書に書かれていることだけを根拠に答え、書かれていない項目や関係を推測で足さないでください。"
    "答えに要る情報が複数の文書に分かれていれば、組み合わせて答えてください。"
    "手順・処理の流れ・項目の列挙を聞かれたら、途中の段階も含めて漏れなく挙げてください。"
    "文書に手がかりがまったく無いときだけ「分かりません」と答えてください。"
    "前置きは省いて簡潔に答えてください。ただし答えの中身は省かないでください。"
    "文書の文言(メッセージなど)を答えるときは、続きの行があれば省略せずに最後まで引用してください。"
    "「図の説明:」で始まる連続した行は、1 つの画面や図の説明が行ごとに分かれたものです。"
    "メッセージや文言を答えるときは、連続する行をつなげて全文を書いてください。"
)

# 資料群(collection)ごとに、どういう資料で、チャンクがどう書かれていて、答えをどう書くかを system prompt に足す。
# 資料群の性質から書き、評価の設問からは逆算しない。質問者が資料群を選んだとき(use_collection)だけ使う。
# system prompt(IMPROVED_PROMPT)は置き換えだが、これは質問ごとに選んだ資料群の分を predict() で末尾に足す
COLLECTION_PROMPTS = {
    "システム開発": (
        "この資料群は、業務システム(Web 画面・バッチ・API)の開発で作られた設計書一式です。"
        "要件定義、方式設計(設計標準・テスト標準)、アプリ設計(機能設計書・テーブル定義書・メッセージ設計書・"
        "外部インタフェース設計書・単体テスト仕様書など)の Excel 文書で、表の行は「列名: 値」の形、"
        "シート上の図や貼られた画面キャプチャの内容は「図の説明:」の行で書かれています。"
        "1 つの機能の仕様が複数の設計書に分かれて書かれていることがあります。"
        "単体テスト仕様書は設計書の内容を確かめる文書なので、設計書に書かれていればそちらを根拠にしてください。"
        "画面 ID・機能 ID・テーブルやカラムの物理名・コード値・メッセージは、該当する行をすべて、文書の表記のまま書いてください。"
        "1 つのメッセージが複数の行に分かれて書かれていることがあります。"
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


# 並べ直しは Decisions API(client.decisions.create)で行う。質問を input に、候補のチャンクを 1 件 1 問として questions に
# 入れ、1 回の呼び出しで全部採点する。チャンクごとに「質問に答えるのにどれだけ役立つか」を 3 段階(RELEVANCE_LEVELS)で問い、
# 返ってきた段階ごとの確率を LEVEL_POINTS の期待値として 0〜10 の点にする。答えの確率だけでなく手がかりの確率も点に入るので、
# 答えが無くても答えのありかを示すチャンクは、無関係なチャンクより上に来る
DECISIONS_MODEL = "gpt-6-luna"  # Decisions API が対応するモデル(これだけ)
RERANK_INPUT = "検索で取り出した文書のチャンクを、質問に答えるのにどれだけ役立つかで採点します。"
RELEVANCE_QUESTION = "このチャンクが、質問に答えるのにどれだけ役立つか。"
RELEVANCE_LEVELS: list[QuestionQuestionParamScoreLevel] = [  # 低い段階から順に並べる(Decisions API の決まり)
    {"label": "無関係", "description": "質問と関係が無い。"},
    {
        "label": "手がかり",
        "description": "答えは無いが、答えを探す手がかり(同じ対象についての別の情報、答えのありかを示す記述)がある。",
    },
    {
        "label": "答え",
        "description": "質問の答えそのもの、またはその一部が書かれている。質問が複数のことを聞いているなら、そのうち 1 つにでも答えていればこれ。",
    },
]
LEVEL_POINTS = {"無関係": 0, "手がかり": 5, "答え": 10}
MAX_QUESTIONS = 400  # Decisions API の 1 回の呼び出しに入れられる問の数の上限
EXPAND_COLUMNS = ["id", "title", "text", "collection", "file", "sheet", "unit", "seq"]


@weave.op
def decisions_create(**kwargs: Any) -> Decision:
    """Decisions API の呼び出し。Weave の OpenAI 自動計装は chat.completions / responses / embeddings だけで decisions を
    対象にしていないので、op にして入出力を記録する。戻り値に model と usage があるので、Weave はそれを使用量として集計し、
    gpt-6-luna の単価で費用を出す(Decisions API は入力トークンだけの課金で単価も同じ)。"""
    return client.decisions.create(**kwargs)


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


_tables: dict[str, lancedb.table.Table] = {}  # 開いたテーブルのキャッシュ(index → テーブル)


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

    質問側と同じ埋め込みモデルで作られていないと検索結果が無意味になるので、メタ情報と突き合わせる
    (別の埋め込みモデルの RagModel から呼ばれることもあるので、キャッシュ済みでも毎回確かめる)。
    """
    built_with = index_meta(index)["embedding_model"]
    if built_with != embedding_model:
        raise RuntimeError(
            f"インデックス {index} は {built_with} で作られています({embedding_model} で検索しようとしている)。"
            f"uv run python -m app.advanced_rag.build_index --extractor {index} --embedding-model {embedding_model} で作り直してください"
        )
    if index not in _tables:
        _tables[index] = lancedb.connect(DB_DIR).open_table(table_name(index))
    return _tables[index]


class RagModel(weave.Model):
    index: str = "structured"  # 検索するインデックス(作ったときの抽出器の名前。basic / structured / vision)
    chat_model: str = "gpt-6-luna"
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 5
    use_collection: bool = False  # true なら検索対象を collection(data/corpus/ 直下のフォルダ = 資料群)に限定する
    system_prompt: str = BASIC_PROMPT
    collection_prompts: dict[str, str] = {}  # collection → system prompt に足す資料群の説明(use_collection のときだけ使う)
    reasoning_effort: ReasoningEffort = "low"
    rank_depth: int = 50  # 診断用に、この順位までの文書 ID を記録する(LLM に渡すのは選んだチャンクだけ)
    search: str = "vector"  # vector(ベクトル検索)/ hybrid(ベクトル検索と全文検索を RRF で混ぜる)
    rrf_k: int = 5  # RRF の k。1 / (k + 順位) の和で並べる。小さいほど各検索の上位が強く効く(よく使われる値は 60)
    rerank: str = "none"  # none / llm(検索の上位 rerank_depth 件を Decisions API で採点し、点の順に並べ直す)
    rerank_depth: int = 20  # 並べ直す候補の数(1 回の呼び出しで採点する)。これより下の順位はそのまま後ろに付ける
    select: str = "top_k"  # top_k(上位 top_k 件)/ score(並べ直しの点が score_threshold 以上。rerank="llm" のときだけ)
    score_threshold: int = 7
    min_selected: int = 1  # score で選ぶとき、閾値以上がこれより少なければ上位から補う
    max_selected: int = 5  # score で選ぶときの上限(広げる前の件数)
    expand: str = "none"  # none / unit(選んだチャンクと同じ区切りのほかのチャンクも渡す)
    max_context_chars: int = 20000  # 広げるときの、渡すチャンクの文字数の合計の上限(選んだチャンクそのものは必ず渡す)

    @weave.op
    def pos_keywords(self, question: str) -> str:
        """全文検索用のキーワード(空白区切り)を、LLM を使わずに形態素の品詞で抜き出す(fulltext.keywords)。"""
        return fulltext.keywords(question)

    @weave.op
    def rerank_scores(self, question: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """候補のチャンクを、Decisions API の 1 回の呼び出しで採点する(チャンク 1 件が 1 問)。

        チャンクごとに、点(score、0〜10。段階ごとの確率の LEVEL_POINTS による期待値)と段階ごとの確率(level_probs)を返す。
        答えられなかった問(refusal)は 0 点にする。
        """
        if not hits:
            return []  # 候補が無い(資料群で絞って 0 件など)ときは API を呼ばない(問 0 件のリクエストはエラーになる)
        if len(hits) > MAX_QUESTIONS:
            raise ValueError(
                f"候補が {len(hits)} 件で Decisions API の問の上限 {MAX_QUESTIONS} を超えます。rerank_depth を減らしてください"
            )
        res = decisions_create(
            model=DECISIONS_MODEL,
            input=f"{RERANK_INPUT}\n\n# 質問\n{question}",
            questions=[
                {
                    "type": "score",
                    "name": f"score_{i}",
                    "instructions": f"{RELEVANCE_QUESTION}\n\n# チャンク: {h['title']}\n{h['text']}",
                    "levels": RELEVANCE_LEVELS,
                }
                for i, h in enumerate(hits)
            ],  # type: ignore[arg-type]
        )
        answers = {a.name: a for a in res.answers}
        judged = []
        for i in range(len(hits)):
            answer = answers[f"score_{i}"]
            probs = {p.label: p.probability for p in answer.probabilities} if answer.type == "score" else {}
            judged.append({"score": sum(LEVEL_POINTS[label] * p for label, p in probs.items()), "level_probs": probs})
        return judged

    def _rerank(self, question: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """上位 rerank_depth 件を Decisions API の点で並べ直す。同点は元の順位の順。score は検索の点から採点の点に置き換える。"""
        head, tail = hits[: self.rerank_depth], hits[self.rerank_depth :]
        judged = self.rerank_scores(question, head)
        order = sorted(range(len(head)), key=lambda i: -judged[i]["score"])  # sorted は安定なので同点は元の順
        return [head[i] | judged[i] for i in order] + tail

    def _select(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """LLM に渡すチャンクを選ぶ。top_k なら上位 top_k 件、score なら並べ直しの点が閾値以上(min〜max 件)。"""
        if self.select == "top_k":
            return hits[: self.top_k]
        if self.select != "score":
            raise ValueError(f"select は top_k / score のどれか: {self.select!r}")
        if self.rerank != "llm":
            raise ValueError("select=score は rerank=llm のときだけ使える(並べ直しの点で選ぶため)")
        scored = hits[: self.rerank_depth]
        chosen = [h for h in scored if h["score"] >= self.score_threshold][: self.max_selected]
        return chosen if len(chosen) >= self.min_selected else scored[: self.min_selected]

    def _expand(self, table: lancedb.table.Table, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """選んだチャンクを、同じ区切り(unit)のほかのチャンクまで広げる。

        選んだチャンクごとに 1 つのまとまり(group)にし、まとまりの中はファイルの中の順(seq)に並べる。
        同じ区切りのチャンクは、選んだチャンクに近い順に max_context_chars まで足す。
        区切りは抽出器が決めたもの(シート、しおりの項目、ページ / スライド)で、ページをまたぐ手順は索引時に 1 つの区切りに
        まとめてある(extractors/continuation.py)ので、ここでは区切りの外には出ない。
        """
        included: set[str] = set()
        size = 0
        groups = []
        for hit in selected:
            if hit["id"] in included:
                continue  # 前のまとまりに入っている
            group = [hit | {"role": "selected"}]
            included.add(hit["id"])
            size += len(hit["text"])
            where = f"file = {quote(hit['file'])} AND unit = {quote(hit['unit'])}"
            unit = table.search().where(where).select(EXPAND_COLUMNS).limit(10000).to_list()
            for row in sorted(unit, key=lambda r: abs(r["seq"] - hit["seq"])):
                if row["id"] in included or size + len(row["text"]) > self.max_context_chars:
                    continue
                group.append(row | {"role": "unit"})
                included.add(row["id"])
                size += len(row["text"])
            groups.append(sorted(group, key=lambda r: r["seq"]))
        return [c | {"group": g} for g, group in enumerate(groups) for c in group]

    def _vector_hits(
        self, table: lancedb.table.Table, question: str, where: str | None, depth: int
    ) -> list[dict[str, Any]]:
        query = embed([question], self.embedding_model)[0]
        # search() の注釈は基底クラスだが、ベクトルを渡すと LanceVectorQueryBuilder が返る(distance_type はそちらにしかない)
        search = cast(LanceVectorQueryBuilder, table.search(query, vector_column_name="vector"))
        search = search.distance_type("cosine").limit(depth)
        if where:
            search = search.where(where, prefilter=True)
        # score はコサイン類似度(1 - cosine 距離)
        return [h | {"score": 1.0 - float(h["_distance"])} for h in search.to_list()]

    def _fts_hits(
        self, table: lancedb.table.Table, question: str, where: str | None, depth: int
    ) -> list[dict[str, Any]]:
        query = fulltext.query(self.pos_keywords(question))
        if query is None:
            return []
        search = table.search(query, query_type="fts").limit(depth)
        if where:
            search = search.where(where, prefilter=True)
        return [h | {"score": float(h["_score"])} for h in search.to_list()]  # score は BM25

    @weave.op
    def retrieve(self, question: str, collection: str | None = None) -> dict[str, Any]:
        """LLM に渡すチャンク(contexts。select と expand で選んだもの)と、上位 rank_depth 件の ID(ranked_ids、正解の順位の診断用)を返す。"""
        table = get_table(self.index, self.embedding_model)
        depth = max(self.top_k, self.rank_depth)
        # 文書の collection 列(メタデータ)で先に絞ってから、その中で検索する(prefilter)
        where = f"collection = {quote(collection)}" if self.use_collection and collection else None
        if self.search == "vector":
            hits = self._vector_hits(table, question, where, depth)
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
                    scores[h["id"]] = scores.get(h["id"], 0.0) + 1.0 / (self.rrf_k + rank)
            hits = [docs[i] | {"score": scores[i]} for i in sorted(scores, key=lambda i: -scores[i])][:depth]
        else:
            raise ValueError(f"search は vector / hybrid のどれか: {self.search!r}")
        if self.rerank == "llm":
            hits = self._rerank(question, hits)
        elif self.rerank != "none":
            raise ValueError(f"rerank は none / llm のどれか: {self.rerank!r}")
        selected = self._select(hits)
        if self.expand == "unit":
            selected = self._expand(table, selected)
        elif self.expand != "none":
            raise ValueError(f"expand は none / unit のどれか: {self.expand!r}")
        drop = {"vector", "_distance", "_score", *fulltext.ALL_COLUMNS}  # 検索用の列と LanceDB が付ける距離・点は渡さない
        contexts = [{k: v for k, v in h.items() if k not in drop} for h in selected]
        return {"contexts": contexts, "ranked_ids": [h["id"] for h in hits]}

    def system_prompt_for(self, collection: str | None) -> str:
        """回答用に LLM に渡す system prompt。資料群の説明があれば、system_prompt の末尾に足す。"""
        if self.use_collection and collection in self.collection_prompts:
            return self.system_prompt + "\n\n# 資料群について\n" + self.collection_prompts[collection]
        return self.system_prompt

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
            g: cs[0]["title"] if len(cs) == 1
            else cs[0]["title"].rpartition(" / ")[0] + " / " + ", ".join(dict.fromkeys(c["unit"] for c in cs))
            for g, cs in blocks.items()
        }
        context_text = "\n\n".join(f"## {titles[g]}\n" + "\n".join(c["text"] for c in cs) for g, cs in blocks.items())
        res = client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "developer", "content": self.system_prompt_for(collection)},
                {"role": "user", "content": f"# 文書\n{context_text}\n\n# 質問\n{question}"},
            ],
            reasoning_effort=self.reasoning_effort,
        )
        return {
            "answer": res.choices[0].message.content,
            "contexts": contexts,
            "ranked_ids": retrieved["ranked_ids"],
        }


if __name__ == "__main__":
    weave.init(PROJECT)
    question = sys.argv[1] if len(sys.argv) > 1 else "プロジェクト登録画面の入力項目は?"
    output = RagModel().predict(question)
    print("Q:", question)
    print("A:", output["answer"])
    for c in output["contexts"]:
        print(f"  - {c['id']} ({c['score']:.3f})")
