from typing import Iterator

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

_prompt_template = '''
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
'''


def stream_rag(query: str, model_name: str, temperature: float) -> Iterator[str]:
    embeddings = init_embeddings(model="text-embedding-3-small", provider="openai")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="./tmp/chroma",
    )

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_template(_prompt_template)

    model = init_chat_model(
        model=model_name, model_provider="openai", temperature=temperature
    )

    documents = retriever.invoke(query)
    chain = prompt | model | StrOutputParser()
    return chain.stream({"question": query, "context": documents})


def app() -> None:
    load_dotenv(override=True)

    with st.sidebar:
        model_name = st.selectbox(
            label="モデル", options=["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
        )
        temperature = st.slider(
            label="temperature", min_value=0.0, max_value=1.0, value=0.0
        )

    st.title("RAG")

    # ユーザーの質問を受け付ける
    question = st.text_input("質問を入力してください")
    if not question:
        return

    # 回答を生成して表示
    stream = stream_rag(question, model_name, temperature)
    st.write_stream(stream)


app()
