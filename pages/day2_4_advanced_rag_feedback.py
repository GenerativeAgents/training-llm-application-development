import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import collect_runs
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langsmith import Client
from pydantic import BaseModel
from streamlit_feedback import streamlit_feedback  # type: ignore[import-untyped]

from app.rag.factory import RAGChainType, create_rag_chain


class SessionState(BaseModel):
    question: str | None
    context: list[Document] | None
    answer: str | None
    run_id: str | None


def show_context(context: list[Document]) -> None:
    st.write("### 検索結果")
    for doc in context:
        source = doc.metadata["source"]
        content = doc.page_content
        with st.expander(source):
            st.text(content)


def app() -> None:
    load_dotenv(override=True)

    # ステートを初期化
    if "state" not in st.session_state:
        st.session_state.state = SessionState(
            question=None,
            context=None,
            answer=None,
            run_id=None,
        )

    with st.sidebar:
        model_name = st.selectbox(label="モデル", options=["gpt-4o-mini", "gpt-4o"])
        temperature = st.slider(
            label="temperature", min_value=0.0, max_value=1.0, value=0.0
        )
        rag_chain_type = st.selectbox(label="RAG Chain Type", options=RAGChainType)

    st.title("Advanced RAG")

    # ユーザーの質問を受け付ける
    question = st.text_input("質問を入力してください")
    if not question:
        return

    # 質問が変わった場合
    if question != st.session_state.state.question:
        st.session_state.state.question = question

        # 回答を生成して表示
        model = ChatOpenAI(model=model_name, temperature=temperature)
        chain = create_rag_chain(rag_chain_type=rag_chain_type, model=model)

        with collect_runs() as cb:
            context_start = False
            answer_start = False
            answer = ""
            for chunk in chain.stream(question):
                if "context" in chunk:
                    if not context_start:
                        context_start = True
                    context = chunk["context"]
                    show_context(context)
                    st.session_state.state.context = context

                if "answer" in chunk:
                    if not answer_start:
                        answer_start = True
                        st.write("### 回答")
                        placeholder = st.empty()

                    answer += chunk["answer"]
                    placeholder.write(answer)

            st.session_state.state.answer = answer
            run_id = cb.traced_runs[0].id
            st.session_state.state.run_id = run_id
    else:
        context = st.session_state.state.context
        show_context(context)
        st.write("### 回答")
        st.write(st.session_state.state.answer)

    # 実行後の場合、フィードバックを受け付ける
    if st.session_state.state.run_id is not None:
        run_id = st.session_state.state.run_id

        feedback = streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="[Optional] Please provide an explanation",
            key=str(run_id),
        )

        if feedback:
            scores = {"👍": 1, "👎": 0}
            score_key = feedback["score"]
            score = scores[score_key]
            comment = feedback.get("text")

            client = Client()
            client.create_feedback(
                run_id=run_id,
                key="thumbs",
                score=score,
                comment=comment,
            )


app()