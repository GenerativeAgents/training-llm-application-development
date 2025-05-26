from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
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


def create_rag_chain(model: BaseChatModel) -> Runnable[str, dict[str, Any]]:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory="./tmp/chroma",
    )

    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_template(_prompt_template)

    return {
        "context": retriever,
        "question": RunnablePassthrough(),
    } | RunnablePassthrough.assign(answer=prompt | model | StrOutputParser())


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
    model = ChatOpenAI(model=model_name, temperature=temperature)
    chain = create_rag_chain(model=model)

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
