import streamlit as st


def app() -> None:
    st.title("Simple Chatbot")

    # ユーザーの入力を受け付ける
    human_message = st.chat_input()
    if not human_message:
        return

    # ユーザーの入力を表示
    with st.chat_message("human"):
        st.write(human_message)

    # 応答を生成して表示
    ai_message = f"{human_message}への応答"
    with st.chat_message("ai"):
        st.write(ai_message)


app()
