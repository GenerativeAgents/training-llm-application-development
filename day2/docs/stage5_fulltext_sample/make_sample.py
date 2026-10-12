"""段階 5(ハイブリッド検索)の全文検索が、文書と質問をどう区切って LanceDB に渡しているかを、実際のデータから書き出す。

LLM は呼ばない。配布済みの LanceDB のテーブル(data/lancedb/docs_vision)と、質問データセット(data/dataset/qa-advance.jsonl)を読む。
既定の題材は、q18「男性の部位別の年齢調整がん死亡率で、1960年ごろに胃の次に高かった部位はどれですか?」と、
その正解の文書(厚労省資料のスライド 36)。1 文字の検索語「胃」があるので、1 文字の列(fts_unigram)が要る理由が見える。

OPENAI_API_KEY があれば、最後にベクトル検索と RRF で混ぜた結果(10_hybrid_rrf.txt)も書く(質問 1 件の埋め込みだけ呼ぶ)。

    uv run python docs/stage5_fulltext_sample/make_sample.py
    uv run python docs/stage5_fulltext_sample/make_sample.py --qa-id q02 --doc-id "<文書の id>" --out <出力先>
"""

import argparse
import json
import os
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import lancedb
from lancedb.index import FTS
from lancedb.query import BooleanQuery, MatchQuery, PhraseQuery

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.advanced_rag import fulltext  # noqa: E402
from app.advanced_rag.rag import DATA_DIR, DB_DIR, RagModel, get_table, quote  # noqa: E402

DEFAULT_QA_ID = "q18"
DEFAULT_DOC_ID = "厚労省資料/健康日本21（第二次）参考資料スライド集#slide36"
DATASET = DATA_DIR / "dataset" / "qa-advance.jsonl"
MORPHEME_LINES = 5  # 文書側の形態素の一覧は、先頭から何行分を書くか


def load_questions() -> dict[str, dict[str, Any]]:
    return {q["qa_id"]: q for q in (json.loads(l) for l in DATASET.open(encoding="utf-8"))}


def morpheme_table(text: str, max_lines: int | None = None) -> str:
    """Sudachi(モード C)の区切りを、行ごとに 表層 / 品詞 / 正規化形 / fts_morph に入る語 の表にする。"""
    out = []
    lines = unicodedata.normalize("NFKC", text).splitlines()
    for line in lines[:max_lines] if max_lines else lines:
        out.append(f"### {line}")
        out.append(f"{'表層':<12} {'品詞':<22} {'正規化形':<12} fts_morph に入る語")
        for m in fulltext._tokenizer().tokenize(line):
            word = m.surface()
            pos = "-".join(p for p in m.part_of_speech()[:3] if p != "*")
            if not fulltext.WORD_CHAR.search(word):
                kept = "(落とす: 記号)"
            else:
                kept = word.lower() if word.isascii() else m.normalized_form().lower()
            out.append(f"{word:<12} {pos:<22} {m.normalized_form():<12} {kept}")
        out.append("")
    return "\n".join(out)


def question_table(question: str) -> str:
    """質問の形態素と、keywords() が残す/落とす判定。"""
    out = [f"質問: {question}", "", f"{'表層':<10} {'品詞':<28} 判定"]
    text = unicodedata.normalize("NFKC", question)
    for m in fulltext._tokenizer().tokenize(text):
        pos = m.part_of_speech()
        label = "-".join(p for p in pos[:3] if p != "*")
        if pos[0] in fulltext.CONTENT_POS:
            note = "残す(内容語)"
            if "副詞可能" in pos:
                note += "。副詞可能なので、単独なら落とす"
            if m.surface() in fulltext.QUESTION_NOUNS:
                note += "。疑問の語なので、単独なら落とす"
        else:
            note = f"落とす({pos[0]})"
        out.append(f"{m.surface():<10} {label:<28} {note}")
    out += ["", f"keywords(): {fulltext.keywords(question)}"]
    return "\n".join(out)


def query_text(keywords: str) -> str:
    """fulltext.query() が組み立てる BooleanQuery を、1 行 1 句で書く。"""
    q = fulltext.query(keywords)
    out = [f"キーワード: {keywords}", "", "BooleanQuery(すべて SHOULD = OR。各句の点を足す)"]
    assert isinstance(q, BooleanQuery)
    for occur, part in q.queries:
        if isinstance(part, MatchQuery):
            out.append(f"  MatchQuery  {part.column:<12} {part.query}   # 語の OR(BM25)")
        elif isinstance(part, PhraseQuery):
            out.append(f"  PhraseQuery {part.column:<12} {part.query}   # 並びの一致で加点")
        else:
            out.append(f"  {type(part).__name__} {part!r}")
    return "\n".join(out)


def is_expected(doc_id: str, question: dict[str, Any]) -> bool:
    """検索結果の id が正解の文書(source_ids)のどれかか。評価(part3_1_eval.ipynb)と同じく前方一致(チャンクに割れた #1, #2 … も当たり)。"""
    return any(doc_id == s or doc_id.startswith(s + "#") for alt in question["source_ids"] for s in alt)


def hits_text(table: lancedb.table.Table, question: dict[str, Any], limit: int = 10) -> str:
    """段階 5 と同じく資料群で絞ってから、全文検索の上位 limit 件を BM25 の点とともに書く。"""
    keywords = fulltext.keywords(question["question"])
    where = f"collection = {quote(question['collection'])}"
    hits = table.search(fulltext.query(keywords), query_type="fts").where(where, prefilter=True).limit(limit).to_list()
    out = [f"{question['qa_id']}: {question['question']}", f"キーワード: {keywords}", f"絞り込み: {where}", "",
           f"{'順位':<4} {'BM25':>8}  id(* は正解の文書)"]
    for n, h in enumerate(hits, 1):
        out.append(f"{n:<4} {h['_score']:>8.2f}  {'*' if is_expected(h['id'], question) else ' '} {h['id']}")
    return "\n".join(out)


def single_char_text(table: lancedb.table.Table, char: str) -> str:
    """1 文字の検索語を bi-gram の列と 1 文字の列で引いた結果の違い。"""
    out = [f"検索語「{char}」を列ごとに MatchQuery で引く", ""]
    for column in (fulltext.COLUMNS["bigram"], fulltext.UNIGRAM):
        hits = table.search(MatchQuery(char, column), query_type="fts").limit(5).to_list()
        out.append(f"{column}: {len(hits)} 件")
        for h in hits:
            out.append(f"  {h['_score']:>6.2f}  {h['id']}")
    return "\n".join(out)


def tokenizer_comparison(table: lancedb.table.Table) -> str:
    """LanceDB 標準のトークナイザ(simple / ngram)で、区切っていない本文に索引を張った場合との比較。一時 DB で試す。"""
    rows = [{"id": r["id"], "text": f"{r['title']}\n{r['text']}"} for r in table.search().limit(table.count_rows()).to_list()]
    out = [f"文書 {len(rows)} 件の title + text(区切っていない文字列)に、LanceDB 標準のトークナイザで FTS インデックスを張って検索した件数(上位 5 件まで表示)", ""]
    with tempfile.TemporaryDirectory() as tmp:
        t = lancedb.connect(tmp).create_table("t", data=rows)
        for label, config, queries in [
            ("simple(既定。空白と句読点で切る)", FTS(), ["胃", "死亡率", "project_name", "年齢調整がん死亡率"]),
            ("ngram(3 文字の n-gram)", FTS(base_tokenizer="ngram"), ["胃", "20", "死亡率", "ER2000123"]),
        ]:
            t.create_index("text", replace=True, config=config)
            out.append(f"## {label}")
            for q in queries:
                hits = t.search(q, query_type="fts").limit(1000).to_list()
                literal = sum(q.lower() in h["text"].lower() for h in hits)
                out.append(f"「{q}」: {len(hits)} 件(そのうち本文に「{q}」を含むもの {literal} 件)")
                for h in hits[:5]:
                    out.append(f"  {h['_score']:>6.2f}  {h['id']}")
            out.append("")
    out.append("## 本リポジトリ(fts_morph + fts_bigram + fts_unigram)")
    for q in ["胃", "死亡率", "project_name", "年齢調整がん死亡率", "ER2000123"]:
        hits = table.search(fulltext.query(q), query_type="fts").limit(1000).to_list()
        literal = sum(fulltext.normalize(q) in fulltext.normalize(h["title"] + "\n" + h["text"]) for h in hits)
        out.append(f"「{q}」: {len(hits)} 件(そのうち本文に「{q}」を含むもの {literal} 件)")
        for h in hits[:5]:
            out.append(f"  {h['_score']:>6.2f}  {h['id']}")
    return "\n".join(out)


def hybrid_text(table: lancedb.table.Table, question: dict[str, Any]) -> str:
    """rag.py の hybrid と同じ計算(上位 depth 件を RRF で混ぜる)を、途中の順位が見える形で書く。"""
    model = RagModel(index="vision", use_collection=True, search="hybrid")
    where = f"collection = {quote(question['collection'])}"
    depth = max(model.top_k, model.rank_depth)
    vec = model._vector_hits(table, question["question"], where, depth)
    fts = model._fts_hits(table, question["question"], where, depth)
    scores: dict[str, float] = {}
    rank_of: dict[str, list[str]] = {}
    for name, hits in (("vector", vec), ("fts", fts)):
        for rank, h in enumerate(hits, 1):
            scores[h["id"]] = scores.get(h["id"], 0.0) + 1.0 / (model.rrf_k + rank)
            rank_of.setdefault(h["id"], []).append(f"{name} {rank} 位")
    merged = sorted(scores, key=lambda i: -scores[i])
    out = [f"{question['qa_id']}: {question['question']}", f"各検索の上位 {depth} 件を RRF(k={model.rrf_k})で混ぜる。RRF = Σ 1 / (k + 順位)", ""]
    for name, hits in (("ベクトル検索(コサイン類似度)", vec), ("全文検索(BM25)", fts)):
        out.append(f"## {name} 上位 5 件")
        for n, h in enumerate(hits[:5], 1):
            out.append(f"{n:<4} {h['score']:>8.3f}  {'*' if is_expected(h['id'], question) else ' '} {h['id']}")
        out.append("")
    out.append(f"## RRF で混ぜた上位 {model.top_k} 件(段階 5 が LLM に渡すチャンク)")
    for n, i in enumerate(merged[: model.top_k], 1):
        out.append(f"{n:<4} {scores[i]:>8.4f}  {'*' if is_expected(i, question) else ' '} {i}   ({' / '.join(rank_of[i])})")
    return "\n".join(out)


def main(qa_id: str, doc_id: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write = lambda name, text: (out / name).write_text(text, encoding="utf-8")  # noqa: E731
    questions = load_questions()
    question = questions[qa_id]
    table = get_table("vision", RagModel.model_fields["embedding_model"].default)

    # (1) DB のレコード(ingest.py が書いた docs_vision.jsonl の 1 件。build_index.py の入力)
    row = table.search().where(f"id = {quote(doc_id)}").limit(1).to_list()[0]
    record = {k: row[k] for k in ("id", "title", "text", "collection", "file", "sheet", "unit", "seq")}
    write("01_db_record.json", json.dumps(record, ensure_ascii=False, indent=2))
    source = f"{row['title']}\n{row['text']}"

    # (2) Sudachi(モード C)の区切り。先頭の数行だけ
    write("02_sudachi_morphemes.txt", morpheme_table(source, MORPHEME_LINES))

    # (3) 3 つの列(fts_morph / fts_bigram / fts_unigram)。テーブルの値と、現行コードの結果が同じことを確かめる
    columns = fulltext.columns(source)
    stored = {c: row[c] for c in fulltext.ALL_COLUMNS}
    if columns != stored:
        print("注意: テーブルの FTS 列は現行コード(fulltext.columns)の結果と違う。テーブルの値を書く")
        columns = stored
    write("03_fts_columns.json", json.dumps({c: columns[c] for c in fulltext.ALL_COLUMNS}, ensure_ascii=False, indent=2))

    # (4) FTS インデックスの設定(build_index.py が渡すもの)と、テーブルに実際に付いているインデックス
    config = dict(base_tokenizer="whitespace", with_position=True, max_token_length=None,
                  lower_case=False, stem=False, remove_stop_words=False, ascii_folding=False)
    indices = [{"name": i.name, "columns": i.columns, "num_indexed_rows": i.num_indexed_rows, "size_bytes": i.size_bytes,
                "index_details": i.index_details} for i in table.list_indices() if i.index_type == "FTS"]
    write("04_index_config.json", json.dumps({"FTS(...)": config, "list_indices()": indices}, ensure_ascii=False, indent=2))

    # (5) 標準のトークナイザで区切らずに索引を張った場合との比較
    write("05_tokenizer_comparison.txt", tokenizer_comparison(table))

    # (6) 質問の形態素と品詞、keywords() の判定
    write("06_question_morphemes.txt", question_table(question["question"]))

    # (7) データセットの全質問のキーワード
    write("07_keywords_all.txt", "\n".join(f"{q['qa_id']}: {q['question']}\n     -> {fulltext.keywords(q['question'])}" for q in questions.values()))

    # (8) 問い合わせ(BooleanQuery)
    write("08_query.txt", query_text(fulltext.keywords(question["question"])))

    # (9) 全文検索の結果と、1 文字の列の効き方
    chars = [t for t in fulltext.bigrams(fulltext.keywords(question["question"])) if len(t) == 1 and not t.isascii()]
    write("09_fts_hits.txt", "\n\n".join([hits_text(table, question), *(single_char_text(table, c) for c in chars[:1])]))

    # (10) ハイブリッド検索(RRF)。質問の埋め込みに OpenAI API を使うので、キーがあるときだけ
    if os.environ.get("OPENAI_API_KEY"):
        write("10_hybrid_rrf.txt", hybrid_text(table, question))
    else:
        print("OPENAI_API_KEY が無いので 10_hybrid_rrf.txt は書かない")
    print(f"-> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--qa-id", default=DEFAULT_QA_ID)
    p.add_argument("--doc-id", default=DEFAULT_DOC_ID)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent))
    a = p.parse_args()
    main(a.qa_id, a.doc_id, Path(a.out))
