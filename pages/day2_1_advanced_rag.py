import streamlit as st
from dotenv import load_dotenv

from app.rag.factory import RAGChainType, create_rag_chain


def app() -> None:
    load_dotenv(override=True)

    with st.sidebar:
        rag_chain_type = st.selectbox(label="RAG Chain Type", options=RAGChainType)

    st.title("Advanced RAG")

    # ユーザーの質問を受け付ける
    question = st.text_input("質問を入力してください")
    if not question:
        return

    # 回答を生成して表示
    chain = create_rag_chain(rag_chain_type=rag_chain_type)

    context_start = False
    answer_start = False
    answer = ""
    for chunk in chain.stream(question):
        if "context" in chunk:
            if not context_start:
                context_start = True
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
