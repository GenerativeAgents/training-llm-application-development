from typing import Iterator

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
)


def stream_llm(messages: list[BaseMessage]) -> Iterator[BaseMessageChunk]:
    model = init_chat_model(
        model="gpt-5-nano",
        model_provider="openai",
        reasoning_effort="minimal",
    )

    all_messages = [SystemMessage(content="You are a helpful assistant.")] + messages
    for chunk in model.stream(all_messages):
        yield chunk


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
    stream = stream_llm(messages)
    with st.chat_message("ai"):
        ai_message = st.write_stream(stream)

    # LLMの応答を会話履歴を追加
    messages.append(AIMessage(content=ai_message))


app()
