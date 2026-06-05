"""WandB Weave データセットアップロードスクリプト.

dataset.yaml を読み込み、WandB Weave にデータセットとして登録する。
同名のデータセットが既に存在する場合は、新しいバージョンが publish される
(Weave のオブジェクトは自動的にバージョニングされる)。

Usage:
    uv run python -m evals.upload_dataset
"""

from pathlib import Path

import weave
import yaml
from dotenv import load_dotenv

from app.weave_init import init_weave

load_dotenv()
init_weave()


DATASET_DIR = Path(__file__).parent
DEFAULT_DATASET_NAME = "inquiry-response-app"


def main() -> None:
    # 1. dataset.yaml 読み込み
    dataset_path = DATASET_DIR / "dataset.yaml"
    with open(dataset_path) as f:
        data = yaml.safe_load(f)
    examples = data["dataset"]
    print(f"dataset.yaml から {len(examples)} 件読み込みました")

    # 2. Weave Dataset 用に行をフラット化
    # target/scorer がデータセット列名をキーワード引数として受け取れるようにする
    rows = []
    for ex in examples:
        row = {
            "customer_name": ex["inputs"]["customer_name"],
            "company_name": ex["inputs"].get("company_name"),
            "content": ex["inputs"]["content"],
            "expected_topic": ex["expected_outputs"]["topic"],
            "expected_response_body": ex["expected_outputs"].get("response_body"),
        }
        rows.append(row)

    # 3. データセットの publish (既存と同名なら新バージョン)
    dataset = weave.Dataset(name=DEFAULT_DATASET_NAME, rows=weave.Table(rows))
    ref = weave.publish(dataset)
    print(f"データセット '{DEFAULT_DATASET_NAME}' に {len(rows)} 件登録しました")
    print(f"Ref: {ref.uri()}")


if __name__ == "__main__":
    main()
