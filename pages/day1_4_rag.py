from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

_prompt_template = '''
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
'''


def create_rag_chain() -> Runnable[Any, str]:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory="./tmp/chroma",
    )

    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_template(_prompt_template)

    model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

    return (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )


def app() -> None:
    load_dotenv(override=True)

    st.title("RAG")

    # ユーザーの質問を受け付ける
    question = st.text_input("質問を入力してください")
    if not question:
        return

    # 回答を生成して表示
    chain = create_rag_chain()
    stream = chain.stream(question)
    st.write_stream(stream)


app()
