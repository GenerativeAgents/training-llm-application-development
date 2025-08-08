import streamlit as st
from dotenv import load_dotenv


def app() -> None:
    load_dotenv(override=True)

    st.title("Indexing")

    clicked = st.button("実行")
    if not clicked:
        return

    st.success("Indexing completed.")


app()
