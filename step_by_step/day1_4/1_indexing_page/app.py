from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI


def create_chain() -> Runnable[dict[str, Any], str]:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder("messages", optional=True),
        ]
    )
    model = ChatOpenAI(model="gpt-4.1-nano")

    return prompt | model | StrOutputParser()


def app() -> None:
    load_dotenv(override=True)

    st.title("Simple Chatbot")

    # 会話履歴を初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    messages: list[BaseMessage] = st.session_state.messages

    # 会話履歴を表示
    for message in messages:
        with st.chat_message(message.type):
            st.write(message.content)

    # ユーザーの入力を受け付ける
    human_message = st.chat_input()
    if not human_message:
        return

    # ユーザーの入力を表示
    with st.chat_message("human"):
        st.write(human_message)

    # ユーザーの入力を会話履歴に追加
    messages.append(HumanMessage(content=human_message))

    # 応答を生成して表示
    chain = create_chain()
    stream = chain.stream({"messages": messages})
    with st.chat_message("ai"):
        ai_message = st.write_stream(stream)

    # LLMの応答を会話履歴を追加
    messages.append(AIMessage(content=ai_message))


app()
