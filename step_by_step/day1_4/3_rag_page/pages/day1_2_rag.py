import streamlit as st
from dotenv import load_dotenv


def app() -> None:
    load_dotenv(override=True)

    st.title("RAG")

    # ユーザーの質問を受け付ける
    question = st.text_input("質問を入力してください")
    if not question:
        return

    st.write(question)


app()
