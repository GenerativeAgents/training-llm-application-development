import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.advanced_rag.factory import chain_constructor_by_name, create_rag_chain


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

    st.title("Advanced RAG")

    # ユーザーの質問を受け付ける
    question = st.text_input("質問を入力してください")
    if not question:
        return

    # 回答を生成して表示
    model = ChatOpenAI(model=model_name, temperature=temperature)
    chain = create_rag_chain(chain_name=chain_name, model=model)

    answer_start = False
    answer = ""
    for chunk in chain.stream(question):
        if "context" in chunk:
            st.write("### 検索結果")
            for doc in chunk["context"]:
                source = doc.metadata["source"]
                content = doc.page_content
                with st.expander(source):
                    st.text(content)

        if "answer" in chunk:
            if not answer_start:
                answer_start = True
                st.write("### 回答")
                placeholder = st.empty()

            answer += chunk["answer"]
            placeholder.write(answer)


app()
