"""検索対象のインデックスを 1 コマンドで作る。毎回作り直す。

1. ingest.py で data/corpus/ を data/lancedb/docs_<抽出器>.jsonl に変換する(--extractor で抽出器を選ぶ)
2. その jsonl を埋め込み、LanceDB のテーブル "docs_<抽出器>" を作る(既にあれば消して作り直す。古いバージョンを残さない)
3. 全文検索用に、タイトルと本文を形態素・bi-gram・1 文字で区切った列を足し、FTS インデックスを張る(fulltext.py)

どの抽出器・埋め込みモデルで作ったかは data/lancedb/docs_<抽出器>.meta.json に残し、eval.py が Weave に記録する。
抽出器ごとに別のテーブルなので、研修の段階を切り替えるたびに作り直す必要は無い(RagModel.index で選ぶ)。
検索側(rag.py)はここで作ったテーブルを開くだけで、無ければエラーになる。

    uv run python -m app.advanced_rag.build_index
    uv run python -m app.advanced_rag.build_index --extractor basic
    uv run python -m app.advanced_rag.build_index --extractor vision            # 図形のあるシートを画像化して Vision LLM に説明させる(VISION_MODEL)
    uv run python -m app.advanced_rag.build_index --embedding-model text-embedding-3-large
"""

import argparse
import json

import lancedb

from . import fulltext
from . import ingest
from .extractors import EXTRACTORS
from .rag import DB_DIR, RagModel, docs_path, embed, load_jsonl, meta_path, table_name


def build(extractor: str, embedding_model: str) -> None:
    meta_path(extractor).unlink(missing_ok=True)  # 作り終えるまでは「無い」扱いにする(rag.has_index)
    ingest.main(extractor)

    docs = load_jsonl(docs_path(extractor))
    vectors = embed([f"{d['title']}\n{d['text']}" for d in docs], embedding_model)
    rows = [{**d, **fulltext.columns(f"{d['title']}\n{d['text']}"), "vector": v.tolist()} for d, v in zip(docs, vectors)]
    db = lancedb.connect(DB_DIR)
    # mode="overwrite" だと古いバージョンのデータが残って膨らむので、テーブルごと消してから作る
    db.drop_table(table_name(extractor), ignore_missing=True)
    table = db.create_table(table_name(extractor), data=rows)
    for column in fulltext.ALL_COLUMNS:
        # 区切りは済んでいるので空白で分けるだけにし、小文字化・語幹・ストップワードは使わない。
        # with_position はフレーズ一致に要る
        table.create_fts_index(
            column, replace=True, base_tokenizer="whitespace", with_position=True, max_token_length=None,
            lower_case=False, stem=False, remove_stop_words=False, ascii_folding=False,
        )
    meta_path(extractor).write_text(json.dumps({"extractor": extractor, "embedding_model": embedding_model}))
    print(f"table {table.name}: {table.count_rows()} rows ({extractor} / {embedding_model}) -> {DB_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", choices=EXTRACTORS, default=ingest.DEFAULT_EXTRACTOR)
    parser.add_argument("--embedding-model", default=RagModel.model_fields["embedding_model"].default)
    args = parser.parse_args()
    build(args.extractor, args.embedding_model)
