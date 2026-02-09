"""
Naive Agent Trajectory 評価スクリプト

エージェントが適切なツールを呼び出すかを評価し、Weave UI で結果を確認する。

使用例:
    # 単一モデルで評価
    uv run python -m evals.day1_5_naive_agent.run_evaluation --model gpt-5-nano

    # 複数モデルで比較評価
    uv run python -m evals.day1_5_naive_agent.run_evaluation --models gpt-5-nano gpt-5-mini

    # reasoning_effortを指定
    uv run python -m evals.day1_5_naive_agent.run_evaluation --model gpt-5-nano --reasoning-effort low

    # カスタムデータセット（Weaveに登録済み）を使用
    uv run python -m evals.day1_5_naive_agent.run_evaluation --model gpt-5-nano --dataset my-dataset
"""

import argparse
import asyncio
import os
from datetime import datetime

import weave
from dotenv import load_dotenv
from weave import Evaluation

from evals.day1_5_naive_agent.predictor import NaiveAgentPredictor
from evals.day1_5_naive_agent.scorers import ToolCallScorer

# デフォルトのインラインデータセット
DEFAULT_DATASET = [
    # Web検索ツールが必要なケース
    {"message": "東京の今日の天気は？", "expected_tool": "tavily_search"},
    {"message": "東京の今日の天気は？Web検索して", "expected_tool": "tavily_search"},
    # LangSmithツールが必要なケース
    {"message": "LangSmithのトレースについて教えて", "expected_tool": "langsmith-retriever"},
    {"message": "LangSmithでプロジェクトを作成する方法は？", "expected_tool": "langsmith-retriever"},
    # LangSmithについてだがWeb検索が必要なケース
    {"message": "LangSmithとWeaveの違いは？", "expected_tool": "tavily_search"},
    {"message": "LangSmithの料金プランは？", "expected_tool": "tavily_search"},
]


async def run_evaluations(
    models: list[str],
    reasoning_effort: str,
    dataset_name: str | None,
):
    """全モデルの評価を単一の非同期コンテキストで実行"""

    weave.init(os.getenv("WEAVE_PROJECT_NAME"))

    # データセットの取得（名前指定がある場合はWeaveから、なければインライン）
    if dataset_name:
        dataset = weave.ref(dataset_name).get()
    else:
        dataset = DEFAULT_DATASET

    scorers = [
        ToolCallScorer(),
    ]

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name} (reasoning_effort={reasoning_effort})")
        print(f"{'='*60}")

        date_str = datetime.now().strftime("%Y%m%d")
        evaluation = Evaluation(
            name=f"naive_agent_{model_name}",
            dataset=dataset,
            scorers=scorers,
            evaluation_name=f"{date_str}_naive_agent_{model_name}_{reasoning_effort}",
        )

        predictor = NaiveAgentPredictor(
            model_name=model_name,
            reasoning_effort=reasoning_effort,
        )
        await evaluation.evaluate(predictor)

    print("\n評価完了。Weave UIで結果を確認してください。")


def main():
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Naive Agent Trajectory 評価")
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
        "--reasoning-effort",
        type=str,
        default="medium",
        choices=["none", "minimal", "low", "medium", "high"],
        help="推論の深さ（デフォルト: medium）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="使用するWeaveデータセット名（省略時はインラインデータセット）",
    )
    args = parser.parse_args()

    # --model または --models のいずれかが必要
    if args.model:
        models = [args.model]
    elif args.models:
        models = args.models
    else:
        # デフォルト: gpt-5-nano のみ
        models = ["gpt-5-nano"]

    asyncio.run(run_evaluations(models, args.reasoning_effort, args.dataset))


if __name__ == "__main__":
    main()
