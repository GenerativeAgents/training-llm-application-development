"""
Documentation Agent 評価スクリプト

複数モデルで Documentation Agent を評価し、Weave UI で性能差を比較する。

使用例:
    # 単一モデル
    uv run python -m evals.day3_documentation_agent.run_evaluation --model gpt-4.1-nano

    # 複数モデル
    uv run python -m evals.day3_documentation_agent.run_evaluation \
      --models gpt-4.1-nano gpt-4.1-mini gpt-4.1

データセットの登録は pages/day3_x_create_documentation_dataset.py を参照
"""

import argparse
import asyncio
import os
from datetime import datetime

import weave
from dotenv import load_dotenv
from weave import Evaluation

from evals.day3_documentation_agent.predictor import DocumentationAgentPredictor
from evals.day3_documentation_agent.scorers import (
    DocumentQualityScorer,
    RequirementsCoverageScorer,
    SectionCompletenessScorer,
)


async def run_evaluations(models: list[str], dataset_name: str):
    """全モデルの評価を単一の非同期コンテキストで実行"""

    weave.init(os.getenv("WEAVE_PROJECT_NAME"))
    dataset = weave.ref(dataset_name).get()

    scorers = [
        SectionCompletenessScorer(),
        RequirementsCoverageScorer(),
        DocumentQualityScorer(),
    ]

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        date_str = datetime.now().strftime("%Y%m%d")
        evaluation = Evaluation(
            name=f"documentation_agent_{model_name}",
            dataset=dataset,
            scorers=scorers,
            evaluation_name=f"{date_str}_documentation_agent_{model_name}",
        )

        predictor = DocumentationAgentPredictor(model_name=model_name)
        await evaluation.evaluate(predictor)

    print("\n評価完了。Weave UIで結果を確認してください。")


def main():
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Documentation Agent 評価")
    parser.add_argument(
        "--model",
        type=str,
        help="評価するモデル（単一）",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="評価するモデル（複数）",
    )
    parser.add_argument(
        "--dataset",
        default="documentation-agent",
        help="使用するデータセット名",
    )
    args = parser.parse_args()

    # --model または --models のいずれかが必要
    if args.model:
        models = [args.model]
    elif args.models:
        models = args.models
    else:
        # デフォルト: gpt-4.1 のみ
        models = ["gpt-4.1"]

    asyncio.run(run_evaluations(models, args.dataset))


if __name__ == "__main__":
    main()
