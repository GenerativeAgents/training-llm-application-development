from typing import Sequence

import streamlit as st
import weave
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from pydantic import BaseModel
from streamlit_feedback import streamlit_feedback  # type: ignore[import-untyped]

from app.advanced_rag.chains.base import AnswerToken, Context, WeaveCallId
from app.advanced_rag.factory import chain_constructor_by_name, create_rag_chain


class SessionState(BaseModel):
    question: str | None
    context: list[Document] | None
    answer: str | None
    weave_call_id: str | None


def show_context(context: Sequence[Document]) -> None:
    st.write("### 検索結果")
    for doc in context:
        source = doc.metadata["source"]
        content = doc.page_content
        with st.expander(source):
            st.text(content)


def app() -> None:
    load_dotenv(override=True)
    weave.init("training-ai-agent-dev")

    # ステートを初期化
    if "state" not in st.session_state:
        st.session_state.state = SessionState(
            question=None,
            context=None,
            answer=None,
            weave_call_id=None,
        )

    with st.sidebar:
        reasoning_effort = st.selectbox(
            label="reasoning_effort",
            options=["minimal", "low", "medium", "high"],
        )
        chain_name = st.selectbox(
            label="RAG Chain Type",
            options=chain_constructor_by_name.keys(),
        )

    st.title("Advanced RAG")

    # ユーザーの質問を受け付ける
    question = st.text_input("質問を入力してください")
    if not question:
        return

    # 質問が変わった場合
    if question != st.session_state.state.question:
        st.session_state.state.question = question

        # 回答を生成して表示
        model = init_chat_model(
            model="gpt-5-nano",
            model_provider="openai",
            reasoning_effort=reasoning_effort,
        )
        chain = create_rag_chain(chain_name=chain_name, model=model)

        answer_start = False
        answer = ""
        for chunk in chain.stream(question):
            if isinstance(chunk, WeaveCallId):
                weave_call_id = chunk.weave_call_id
                st.session_state.state.weave_call_id = weave_call_id

            if isinstance(chunk, Context):
                context = chunk.documents
                show_context(context)
                st.session_state.state.context = context

            if isinstance(chunk, AnswerToken):
                if not answer_start:
                    answer_start = True
                    st.write("### 回答")
                    placeholder = st.empty()

                answer += chunk.token
                placeholder.write(answer)

            st.session_state.state.answer = answer
    else:
        context = st.session_state.state.context
        show_context(context)
        st.write("### 回答")
        st.write(st.session_state.state.answer)

    # 実行後の場合、フィードバックを受け付ける
    if st.session_state.state.weave_call_id is not None:
        weave_call_id = st.session_state.state.weave_call_id

        feedback = streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="[Optional] Please provide an explanation",
            key=str(weave_call_id),
        )

        if feedback:
            scores = {"👍": 1, "👎": 0}
            score_key = feedback["score"]
            score = scores[score_key]
            comment = feedback.get("text")

            client = weave.init("training-ai-agent-dev")
            call = client.get_call(weave_call_id)

            call.feedback.add_reaction(score_key)
            call.feedback.add("score", {"value": score})
            if comment:
                call.feedback.add_note(comment)


app()
