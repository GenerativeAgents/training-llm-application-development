"""
Advanced RAG Chain Type 評価スクリプト

7種類のRAG Chain（naive, hyde, multi_query, rag_fusion, rerank, route, hybrid）を
一括評価し、Weave UIで性能差を比較する。

使用例:
    uv run python -m evals.day2_advanced_rag.run_evaluation
    uv run python -m evals.day2_advanced_rag.run_evaluation --chains naive hyde
    uv run python -m evals.day2_advanced_rag.run_evaluation --model gpt-4.1-nano

データセットの登録は pages/day2_x_create_dataset.py を参照
"""

import argparse
import asyncio
import os
from datetime import datetime
from typing import Any

import weave
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from weave import Evaluation, Model
from weave.scorers import ContextEntityRecallScorer, HallucinationFreeScorer

from app.advanced_rag.chains.base import AnswerToken, Context
from app.advanced_rag.factory import chain_constructor_by_name, create_rag_chain

CHAIN_TYPES = list(chain_constructor_by_name.keys())


class RAGPredictor(Model):
    chain_name: str
    model_name: str

    @weave.op
    def predict(self, question: str) -> dict[str, Any]:
        model = init_chat_model(model=self.model_name, model_provider="openai")
        chain = create_rag_chain(chain_name=self.chain_name, model=model)

        context: list[Document] = []
        answer = ""
        for chunk in chain.stream(question):
            if isinstance(chunk, Context):
                context.extend(chunk.documents)
            if isinstance(chunk, AnswerToken):
                answer += chunk.token

        # スコアラー用にcontextを文字列化
        context_str = "\n".join([doc.page_content for doc in context])

        return {
            "context": context,
            "context_str": context_str,
            "answer": answer,
        }


class ContextRecallScorer(weave.Scorer):
    """
    ContextEntityRecallScorerのラッパー

    期待回答のエンティティが取得コンテキストに含まれるかを評価

    パラメータマッピング:
        - output: 期待する回答（dataset.answer）
        - context: 取得したコンテキスト（prediction.context_str）
    """

    model_id: str = "openai/gpt-4.1-nano"

    @weave.op
    async def score(self, output: dict[str, Any], answer: str) -> dict:
        scorer = ContextEntityRecallScorer(model_id=self.model_id)
        result = await scorer.score(
            output=answer,  # 期待する回答
            context=output["context_str"],  # 取得したコンテキスト
        )
        return {
            "recall": result["recall"],
        }


class HallucinationScorer(weave.Scorer):
    """
    HallucinationFreeScorerのラッパー

    生成された回答がコンテキストに基づいているか（ハルシネーションがないか）を評価

    パラメータマッピング:
        - output: 生成された回答（prediction.answer）
        - context: 取得したコンテキスト（prediction.context_str）
    """

    model_id: str = "openai/gpt-4.1-nano"

    @weave.op
    async def score(self, output: dict[str, Any]) -> dict:
        scorer = HallucinationFreeScorer(model_id=self.model_id)
        result = await scorer.score(
            output=output["answer"],  # 生成された回答
            context=output["context_str"],  # 取得したコンテキスト
        )
        return {
            "hallucination_free": not result["has_hallucination"],
            "score": 0 if result["has_hallucination"] else 1,
            "reasoning": result.get("conclusion", ""),
        }


async def run_evaluations(chains: list[str], model_name: str, dataset_name: str):
    """全チェーンの評価を単一の非同期コンテキストで実行"""

    weave.init(os.getenv("WEAVE_PROJECT_NAME"))
    dataset = weave.ref(dataset_name).get()
    scorers = [ContextRecallScorer(), HallucinationScorer()]

    for chain_name in chains:
        print(f"\n{'='*60}")
        print(f"Evaluating: {chain_name}")
        print(f"{'='*60}")

        date_str = datetime.now().strftime("%Y%m%d")
        evaluation = Evaluation(
            name=f"{model_name}_{chain_name}",  # Objectsタブでの識別名
            dataset=dataset,
            scorers=scorers,
            evaluation_name=f"{date_str}_rag_{model_name}_{chain_name}",  # Trace表示名
        )

        predictor = RAGPredictor(
            chain_name=chain_name,
            model_name=model_name,
        )
        await evaluation.evaluate(predictor)

    print("評価完了。")


def main():
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Advanced RAG評価")
    parser.add_argument("--chains", nargs="+", default=CHAIN_TYPES, choices=CHAIN_TYPES)
    parser.add_argument("--model", default="gpt-4.1-nano")
    parser.add_argument("--dataset", default="training-llm-app")
    args = parser.parse_args()

    asyncio.run(run_evaluations(args.chains, args.model, args.dataset))


if __name__ == "__main__":
    main()
