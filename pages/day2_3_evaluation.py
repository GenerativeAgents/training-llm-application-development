import time
from typing import Any

import nest_asyncio
import streamlit as st
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision
from ragas.metrics.base import MetricWithEmbeddings, MetricWithLLM, SingleTurnMetric

from app.rag.factory import RAGChainType, create_rag_chain


class RagasMetricEvaluator:
    def __init__(
        self,
        metric: SingleTurnMetric,
        llm: BaseChatModel,
        embeddings: Embeddings,
    ):
        self.metric = metric

        # LLMとEmbeddingsをMetricに設定
        if isinstance(self.metric, MetricWithLLM):
            self.metric.llm = LangchainLLMWrapper(llm)
        if isinstance(self.metric, MetricWithEmbeddings):
            self.metric.embeddings = LangchainEmbeddingsWrapper(embeddings)

    def __call__(self, run: Run, example: Example) -> dict[str, Any]:
        if run.outputs is None:
            raise ValueError("run.outputs is None.")
        if example.outputs is None:
            raise ValueError("example.outputs is None.")

        sample = SingleTurnSample(
            user_input=example.inputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=[doc.page_content for doc in run.outputs["contexts"]],
            reference=example.outputs["ground_truth"],
        )
        score = self.metric.single_turn_score(sample)
        return {"key": self.metric.name, "score": score}


class Predictor:
    def __init__(self, chain: Runnable[str, dict[str, Any]]):
        self.chain = chain

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs["question"]
        output = self.chain.invoke(question)
        return {
            "contexts": output["context"],
            "answer": output["answer"],
        }


def app() -> None:
    with st.sidebar:
        rag_chain_type = st.selectbox(label="RAG Chain Type", options=RAGChainType)

    st.title("Evaluation")

    clicked = st.button("実行")
    if not clicked:
        return

    metrics = [context_precision, answer_relevancy]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    evaluators = [RagasMetricEvaluator(metric, llm, embeddings) for metric in metrics]

    nest_asyncio.apply()

    with st.spinner("Evaluating..."):
        start_time = time.time()

        chain = create_rag_chain(rag_chain_type=rag_chain_type)
        predictor = Predictor(chain=chain)

        evaluate(
            predictor,
            data="training-llm-app",
            evaluators=evaluators,
            metadata={"rag_chain_type": rag_chain_type},
        )

        end_time = time.time()

    elapsed_time = end_time - start_time
    st.success(f"Evaluation completed. Elapsed time: {elapsed_time:.2f} sec.")


app()
