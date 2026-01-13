import streamlit as st
import weave
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from app.advanced_rag.chains.base import AnswerToken, Context
from app.advanced_rag.factory import chain_constructor_by_name, create_rag_chain


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

    st.title("Advanced RAG")

    # ユーザーの質問を受け付ける
    question = st.text_input("質問を入力してください")
    if not question:
        return

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
        if isinstance(chunk, Context):
            st.write("### 検索結果")
            for doc in chunk.documents:
                source = doc.metadata["source"]
                content = doc.page_content
                with st.expander(source):
                    st.text(content)

        if isinstance(chunk, AnswerToken):
            if not answer_start:
                answer_start = True
                st.write("### 回答")
                placeholder = st.empty()

            answer += chunk.token
            placeholder.write(answer)


app()
