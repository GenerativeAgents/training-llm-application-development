import time
from typing import Any, Type

import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextPrecision, ResponseRelevancy
from ragas.metrics.base import MetricWithEmbeddings, MetricWithLLM, SingleTurnMetric

from app.rag.factory import chain_constructor_by_name, create_rag_chain


class RagasMetricEvaluator:
    def __init__(
        self,
        metric_class: Type[SingleTurnMetric],
        llm: BaseChatModel,
        embeddings: Embeddings,
    ):
        self.metric = metric_class()

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
    load_dotenv(override=True)

    with st.sidebar:
        model_name = st.selectbox(label="モデル", options=["gpt-4o-mini", "gpt-4o"])
        temperature = st.slider(
            label="temperature", min_value=0.0, max_value=1.0, value=0.0
        )
        chain_name = st.selectbox(
            label="RAG Chain Type",
            options=chain_constructor_by_name.keys(),
        )

    st.title("Evaluation")

    clicked = st.button("実行")
    if not clicked:
        return

    metrics = [ContextPrecision, ResponseRelevancy]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    evaluators = [RagasMetricEvaluator(metric, llm, embeddings) for metric in metrics]

    nest_asyncio.apply()

    with st.spinner("Evaluating..."):
        start_time = time.time()

        model = ChatOpenAI(model=model_name, temperature=temperature)
        chain = create_rag_chain(chain_name=chain_name, model=model)
        predictor = Predictor(chain=chain)

        evaluate(
            predictor,
            data="training-llm-app",
            evaluators=evaluators,
            metadata={
                "model_name": model_name,
                "temperature": temperature,
                "chain_name": chain_name,
            },
        )

        end_time = time.time()

    elapsed_time = end_time - start_time
    st.success(f"Evaluation completed. Elapsed time: {elapsed_time:.2f} sec.")


app()
