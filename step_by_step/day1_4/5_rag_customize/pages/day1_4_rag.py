from typing import Iterator

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

_prompt_template = '''
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
'''


def reduce_fn(outputs):
    """ストリーミング出力をLangSmithのトレースエントリで1つにまとめる"""
    return "".join(str(chunk.content) for chunk in outputs)


@traceable(run_type="chain", reduce_fn=reduce_fn)
def stream_rag(query: str, reasoning_effort: str) -> Iterator[BaseMessageChunk]:
    embeddings = init_embeddings(model="text-embedding-3-small", provider="openai")
    vector_store = Chroma(
        embedding_function=embeddings,  # type: ignore[arg-type]
        persist_directory="./tmp/chroma",
    )

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_template(_prompt_template)

    model = init_chat_model(
        model="gpt-5-nano",
        model_provider="openai",
        reasoning_effort=reasoning_effort,
    )

    documents = retriever.invoke(query)
    prompt_value = prompt.invoke({"question": query, "context": documents})
    return model.stream(prompt_value)


def app() -> None:
    load_dotenv(override=True)

    with st.sidebar:
        reasoning_effort = st.selectbox(
            label="reasoning_effort",
            options=["minimal", "low", "medium", "high"],
        )

    st.title("RAG")

    # ユーザーの質問を受け付ける
    question = st.text_input("質問を入力してください")
    if not question:
        return

    # 回答を生成して表示
    stream = stream_rag(question, reasoning_effort)
    st.write_stream(stream)


app()
