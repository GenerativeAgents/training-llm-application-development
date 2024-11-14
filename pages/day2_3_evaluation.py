from typing import Any

import nest_asyncio
import streamlit as st
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision
from ragas.metrics.base import Metric, MetricWithEmbeddings, MetricWithLLM

from core.rag.naive import create_naive_rag_chain


class RagasMetricEvaluator:
    def __init__(self, metric: Metric, llm: BaseChatModel, embeddings: Embeddings):
        self.metric = metric

        # LLMとEmbeddingsをMetricに設定
        if isinstance(self.metric, MetricWithLLM):
            self.metric.llm = LangchainLLMWrapper(llm)
        if isinstance(self.metric, MetricWithEmbeddings):
            self.metric.embeddings = LangchainEmbeddingsWrapper(embeddings)

    def evaluate(self, run: Run, example: Example) -> dict[str, Any]:
        context_strs = [doc.page_content for doc in run.outputs["contexts"]]

        # Ragasの評価メトリクスのscoreメソッドでスコアを算出
        score = self.metric.score(
            {
                "question": example.inputs["question"],
                "answer": run.outputs["answer"],
                "contexts": context_strs,
                "ground_truth": example.outputs["ground_truth"],
            },
        )
        return {"key": self.metric.name, "score": score}


def predict(inputs: dict[str, Any]) -> dict[str, Any]:
    question = inputs["question"]
    chain = create_naive_rag_chain()
    output = chain.invoke(question)
    return {
        "contexts": output["context"],
        "answer": output["answer"],
    }


def app() -> None:
    st.title("Evaluation")

    clicked = st.button("実行")
    if not clicked:
        return

    metrics = [context_precision, answer_relevancy]

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    evaluators = [
        RagasMetricEvaluator(metric, llm, embeddings).evaluate for metric in metrics
    ]

    nest_asyncio.apply()

    with st.spinner("Evaluating..."):
        evaluate(
            predict,
            data="training-llm-app",
            evaluators=evaluators,
        )
    st.success("Evaluation completed.")


app()
