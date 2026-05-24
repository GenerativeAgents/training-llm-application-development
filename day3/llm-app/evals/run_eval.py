"""WandB Weave 評価実行スクリプト.

Usage:
    uv run python -m evals.run_eval
"""

import asyncio
from typing import Any

import weave
from dotenv import load_dotenv

from app.generate.graph import graph
from evals.evaluators.classification_accuracy import classification_accuracy
from evals.evaluators.forbidden_content_judge import forbidden_content_judge
from evals.evaluators.politeness_judge import politeness_judge

load_dotenv()
weave.init("training-ai-agent-dev")

DEFAULT_DATASET_NAME = "inquiry-response-app"


@weave.op()
async def target(
    customer_name: str,
    content: str,
    company_name: str | None = None,
) -> dict[str, Any]:
    """LangGraph ワークフローを実行するターゲット関数."""
    result: dict[str, Any] = await graph.ainvoke(
        {
            "content": content,
            "customer_name": customer_name,
            "company_name": company_name,
        }
    )
    return result


async def main() -> None:
    print(f"データセット: {DEFAULT_DATASET_NAME}")
    print("評価を開始します...")

    dataset = weave.ref(DEFAULT_DATASET_NAME).get()
    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[
            classification_accuracy,
            politeness_judge,
            forbidden_content_judge,
        ],
    )

    results = await evaluation.evaluate(target)

    print("評価完了")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
