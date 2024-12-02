import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def app() -> None:
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
    ai_message = f"{human_message}への応答"
    with st.chat_message("ai"):
        st.write(ai_message)

    # LLMの応答を会話履歴を追加
    messages.append(AIMessage(content=ai_message))


app()
