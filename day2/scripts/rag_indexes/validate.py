"""配布前に3テーブルの検索と生成条件を検証する。外部APIは呼ばない。"""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import lancedb
from lancedb.query import MatchQuery

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from app.advanced_rag import fulltext  # noqa: E402

INDEXES = ("basic", "structured", "vision")
PACKAGES = ("lancedb", "pyarrow", "sudachipy", "sudachidict-core")
DB_DIR = PROJECT / "data/lancedb"


def validate(db_dir: Path) -> dict:
    db = lancedb.connect(db_dir)
    indexes = {}
    for index in INDEXES:
        name = f"docs_{index}"
        with (db_dir / f"{name}.jsonl").open() as source:
            records = [json.loads(line) for line in source]
        if not records or len({record["id"] for record in records}) != len(records):
            raise ValueError(f"{name}: 文書が空かIDが重複しています")
        meta = json.loads((db_dir / f"{name}.meta.json").read_text())
        if meta["extractor"] != index:
            raise ValueError(f"{name}: 抽出器の記録が一致しません")
        table = db.open_table(name)
        expected = {*records[0], "vector", *fulltext.ALL_COLUMNS}
        if not expected.issubset(table.schema.names) or table.count_rows() != len(
            records
        ):
            raise ValueError(f"{name}: スキーマまたは文書数が一致しません")
        indexed = {column for item in table.list_indices() for column in item.columns}
        if not set(fulltext.ALL_COLUMNS).issubset(indexed):
            raise ValueError(f"{name}: 全文検索インデックスが不足しています")
        rows = table.search().limit(10).to_list()
        first = rows[0]
        where = "collection = '" + first["collection"].replace("'", "''") + "'"
        vector = (
            table.search(first["vector"], query_type="vector")
            .where(where)
            .limit(1)
            .to_list()
        )
        if not vector or vector[0]["collection"] != first["collection"]:
            raise ValueError(f"{name}: ベクトル検索・絞り込みに失敗しました")
        for column in fulltext.ALL_COLUMNS:
            sample = next((row for row in rows if row[column].strip()), None)
            if sample is None:
                # 先頭に英語資料しか無い場合も、日本語の列に語がある行で試す。
                found = table.search().where(f"{column} != ''").limit(1).to_list()
                sample = found[0] if found else None
            if (
                sample is None
                or not table.search(
                    MatchQuery(sample[column].split()[0], column), query_type="fts"
                )
                .limit(1)
                .to_list()
            ):
                raise ValueError(f"{name}: {column} の全文検索に失敗しました")
        indexes[index] = {
            **meta,
            "rows": len(records),
            "vector_dimensions": len(first["vector"]),
        }
        print(f"検証OK: {name} ({len(records)}件、ベクトル・全文検索・絞り込み)")
    return indexes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, default=DB_DIR)
    parser.add_argument("--source-sha")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    indexes = validate(args.db_dir)
    if args.output:
        if not args.source_sha:
            parser.error("--output には --source-sha が必要です")
        # rag.py同様のdotenv読込を行うが、APIクライアントやWeaveは初期化しない。
        from dotenv import load_dotenv

        load_dotenv(PROJECT / ".env", override=True)
        from app.advanced_rag.vision_config import VISION_MODEL, VISION_REASONING_EFFORT

        manifest = {
            "format_version": 1,
            "source_sha": args.source_sha,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
            "packages": {name: version(name) for name in PACKAGES},
            "indexes": indexes,
            "vision_model": VISION_MODEL,
            "vision_reasoning_effort": VISION_REASONING_EFFORT,
        }
        args.output.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
