import asyncio
import time
from typing import Any

import streamlit as st
import weave
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from weave import Evaluation, Model

from app.advanced_rag.chains.base import AnswerToken, BaseRAGChain, Context
from app.advanced_rag.factory import chain_constructor_by_name, create_rag_chain

# RAGの処理を呼び出す関数 (クラス)


class Predictor(Model):
    chain: BaseRAGChain

    @weave.op
    def predict(self, question: str) -> dict[str, Any]:
        context: list[Document] = []
        answer = ""

        for chunk in self.chain.stream(question):
            if isinstance(chunk, Context):
                context.extend(chunk.documents)

            if isinstance(chunk, AnswerToken):
                answer += chunk.token

        return {
            "context": context,
            "answer": answer,
        }


# 検索の評価器 (Evaluator)


@weave.op
def context_recall(output: dict[str, Any], context: str) -> int:
    """
    想定する情報源のうち、検索結果に含まれた割合を算出します。
    用意しているデータセットでは想定する情報源は1件のみのため、
    検索結果に想定する情報源が含まれる場合は1、含まれない場合は0、
    というスコアになります。
    """
    output_context: list[Document] = output["context"]
    search_result_sources: list[str] = [r.metadata["source"] for r in output_context]
    ground_truch_source: str = context

    if ground_truch_source in search_result_sources:
        score = 1
    else:
        score = 0
    return score


# 生成の評価器 (Evaluator)


class AnswerHallucinationOutput(BaseModel):
    reasoning: str
    hallucination: bool = Field(
        description="TRUE if the output contains any hallucinations (unsupported claims, contradictions, speculative details, or inaccurate facts). FALSE if all claims are directly verifiable from the input context."
    )


# 以下のプロンプトは、LangSmithが提供するOnline Evaluatorのプロンプトを日本語訳したものです
_answer_hallucination_prompt = """
あなたは、モデル出力の幻覚（ハルシネーション）を評価する専門的なデータラベラーです。以下のルーブリックに基づいてスコアを割り当てることがあなたのタスクです：

<Rubric>
  幻覚のない回答とは：
  - 入力コンテキストによって直接裏付けられる検証可能な事実のみを含む
  - 裏付けのない主張や仮定を行わない
  - 推測的または想像上の詳細を追加しない
  - 日付、数字、および具体的な詳細において完全な正確性を保つ
  - 情報が不完全な場合は適切に不確実性を示す
</Rubric>

<Instructions>
  - 入力コンテキストを徹底的に読む
  - 出力で行われているすべての主張を特定する
  - 各主張を入力コンテキストと相互参照する
  - 裏付けのない情報や矛盾する情報を記録する
  - 幻覚の重大性と数量を考慮する
</Instructions>

<Reminder>
  事実の正確性と入力コンテキストからの裏付けのみに焦点を当ててください。採点においてスタイル、文法、またはプレゼンテーションは考慮しないでください。短くても事実に基づく回答は、裏付けのない主張を含む長い回答よりも高いスコアを付けるべきです。
</Reminder>

出力の幻覚を評価するために、以下のコンテキストを使用してください：
<context>
{context}
</context>

<input>
{input}
</input>

<output>
{output}
</output>

利用可能な場合は、以下の参照出力も回答の幻覚を特定するのに役立てることができます：
<reference_outputs>
{reference_outputs}
</reference_outputs>
""".strip()


@weave.op
def answer_hallucination(output: dict[str, Any], question: str, answer: str) -> int:
    prompt = ChatPromptTemplate.from_template(_answer_hallucination_prompt)
    model = init_chat_model(
        model="gpt-5-nano",
        model_provider="openai",
        reasoning_effort="minimal",
    )
    model_with_structure = model.with_structured_output(AnswerHallucinationOutput)

    prompt_value = prompt.invoke(
        {
            "input": question,
            "context": output["context"],
            "output": output["answer"],
            "reference_outputs": answer,
        }
    )
    output: AnswerHallucinationOutput = model_with_structure.invoke(prompt_value)  # type: ignore[assignment]

    # ハルシネーションのある場合は0、ない場合は1を返す
    if output.hallucination:
        score = 0
    else:
        score = 1
    return score


# Streamlitのアプリ


def app() -> None:
    load_dotenv(override=True)
    weave.init("training-llm-app")

    with st.sidebar:
        reasoning_effort = st.selectbox(
            label="reasoning_effort",
            options=["minimal", "low", "medium", "high"],
        )
        chain_name = st.selectbox(
            label="RAG Chain Type",
            options=chain_constructor_by_name.keys(),
        )

    st.title("Evaluation")

    dataset_ref = weave.ref("dataset-example").get()
    dataset_df = dataset_ref.to_pandas()
    st.write(dataset_df)

    clicked = st.button("実行")
    if not clicked:
        return

    with st.spinner("Evaluating..."):
        start_time = time.time()

        # 推論の準備
        model = init_chat_model(
            model="gpt-5-nano",
            model_provider="openai",
            reasoning_effort=reasoning_effort,
        )
        chain = create_rag_chain(chain_name=chain_name, model=model)
        predictor = Predictor(chain=chain)

        # 評価の実行
        evaluation = Evaluation(
            dataset=dataset_ref,
            scorers=[context_recall, answer_hallucination],
        )
        asyncio.run(evaluation.evaluate(predictor))

        end_time = time.time()

    elapsed_time = end_time - start_time
    st.success(f"Evaluation completed. Elapsed time: {elapsed_time:.2f} sec.")


app()
