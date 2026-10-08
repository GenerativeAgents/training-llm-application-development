"""data/corpus/ 配下のオフィスファイル(xlsx / pdf / pptx)を検索単位に分割して data/lancedb/docs_<抽出器>.jsonl に書き出す。

通常は build_index.py から呼ばれる(取り込み → 埋め込み → LanceDB まで 1 コマンド)。
抽出結果だけ確認したいときは単体でも実行できる。

テキストの取り出し方は extractors/ の抽出器で切り替える(--extractor)。抽出器はファイルを
位置単位(シート / スライド / ページ)に分け、長い位置単位をチャンクに切るところまでを担当する
(切り方は抽出器ごとに変えられる。共通の切り方は extractors/chunking.py)。
ここではチャンクに ID・タイトル・メタ情報を付けて書き出すだけ。
ID は `corpus からの相対パス(拡張子なし)#シート名(#連番)` で、抽出器によらず同じ規則
(正解データの source_ids と前方一致で照合するため、抽出器には決めさせない)。
別のフォルダに同名のファイルがあっても衝突しない。

    uv run python -m app.advanced_rag.ingest
    uv run python -m app.advanced_rag.ingest --extractor basic
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .extractors import EXTRACTORS, SUFFIXES
from .rag import DATA_DIR, docs_path

CORPUS_DIR = DATA_DIR / "corpus"
DEFAULT_EXTRACTOR = "structured"


def ingest_file(path: Path, extractor: str) -> list[dict[str, Any]]:
    rel = path.relative_to(CORPUS_DIR)
    collection = rel.parts[0]  # 資料群(data/corpus/ 直下のフォルダ)。検索時の絞り込みに使う
    doc_id = rel.with_suffix("").as_posix()
    doc_title = " / ".join(rel.with_suffix("").parts)  # フォルダ階層も含める(同名ファイルを区別できるように)
    records: list[dict[str, Any]] = []
    for sheet, chunks, *meta in EXTRACTORS[extractor](path):
        for i, text in enumerate(chunks):
            suffix = f"#{i + 1}" if len(chunks) > 1 else ""
            extra = meta[0][i] if meta else {}
            record = {
                "id": f"{doc_id}#{sheet}{suffix}",
                "title": f"{doc_title} / {sheet}",
                "text": text,
                "collection": collection,
                "file": str(rel),
                "sheet": sheet,
                "unit": extra.get("unit", sheet),  # チャンクを切り出した区切り(PDF のしおりの項目なら項目全体のページ範囲)
                "seq": len(records),  # ファイルの中でのチャンクの順番(前後のチャンクを引くのに使う)
            }
            records.append(record)
    return records


def main(extractor: str = DEFAULT_EXTRACTOR) -> None:
    paths = sorted(p for p in CORPUS_DIR.rglob("*") if p.suffix in SUFFIXES and not p.name.startswith("~$"))
    records = [r for p in paths for r in ingest_file(p, extractor)]
    ids = Counter(r["id"] for r in records)
    dupes = sorted(i for i, n in ids.items() if n > 1)
    if dupes:
        sys.exit(f"ID が重複しています: {dupes}")
    out_path = docs_path(extractor)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sizes = [len(r["text"]) for r in records]
    print(f"[{extractor}] {len(paths)} files -> {len(records)} docs -> {out_path}")
    print(f"chars/doc: min={min(sizes)} median={sorted(sizes)[len(sizes) // 2]} max={max(sizes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", choices=EXTRACTORS, default=DEFAULT_EXTRACTOR)
    main(parser.parse_args().extractor)
