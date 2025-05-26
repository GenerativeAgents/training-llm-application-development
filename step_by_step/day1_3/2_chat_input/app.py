import streamlit as st


def app() -> None:
    st.title("Simple Chatbot")

    # ユーザーの入力を受け付ける
    human_message = st.chat_input()
    print(human_message)


app()
