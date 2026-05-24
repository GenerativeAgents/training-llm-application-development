"""LangSmith 評価実行スクリプト.

Usage:
    uv run python -m evals.run_eval
"""

import asyncio
from typing import Any

from dotenv import load_dotenv
from langsmith.evaluation import aevaluate

from app.generate.graph import graph
from evals.evaluators.classification_accuracy import classification_accuracy
from evals.evaluators.forbidden_content_judge import forbidden_content_judge
from evals.evaluators.politeness_judge import politeness_judge

load_dotenv()

DEFAULT_DATASET_NAME = "llm-app-evals-book"


async def target(inputs: dict[str, Any]) -> dict[str, Any]:
    """LangGraph ワークフローを実行するターゲット関数."""
    result: dict[str, Any] = await graph.ainvoke(
        {
            "content": inputs["content"],
            "customer_name": inputs["customer_name"],
            "company_name": inputs.get("company_name"),
        }
    )
    return result


async def main() -> None:
    print(f"データセット: {DEFAULT_DATASET_NAME}")
    print("評価を開始します...")

    results = await aevaluate(
        target,
        data=DEFAULT_DATASET_NAME,
        evaluators=[
            classification_accuracy,
            politeness_judge,
            forbidden_content_judge,
        ],
    )

    print(f"評価完了: {results.experiment_name}")


if __name__ == "__main__":
    asyncio.run(main())
