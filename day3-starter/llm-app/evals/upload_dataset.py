"""LangSmith データセットアップロードスクリプト.

dataset.yaml を読み込み、LangSmith にデータセットとして登録する。

Usage:
    uv run python -m evals.upload_dataset
"""

from pathlib import Path

import yaml
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()


DATASET_DIR = Path(__file__).parent
DEFAULT_DATASET_NAME = "llm-app-evals-book"


def main() -> None:
    # 1. dataset.yaml 読み込み
    dataset_path = DATASET_DIR / "dataset.yaml"
    with open(dataset_path) as f:
        data = yaml.safe_load(f)
    examples = data["dataset"]
    print(f"dataset.yaml から {len(examples)} 件読み込みました")

    # 2. データセットの作成 (既存データセットは削除して作り直す)
    client = Client()
    dataset_name = DEFAULT_DATASET_NAME
    if client.has_dataset(dataset_name=dataset_name):
        print(f"  既存データセット '{dataset_name}' を削除します...")
        client.delete_dataset(dataset_name=dataset_name)
    dataset = client.create_dataset(dataset_name=dataset_name)

    # 3. 評価データ (Examples) の登録
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {"inputs": ex["inputs"], "outputs": ex["expected_outputs"]}
            for ex in examples
        ],
    )
    print(f"  データセット '{dataset_name}' に {len(examples)} 件登録しました")
    print("完了")


if __name__ == "__main__":
    main()
