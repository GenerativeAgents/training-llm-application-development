import time
from typing import Generator

import streamlit as st

response_message = "Streamlitのセットアップが完了しました！"


def generate_stream_response() -> Generator[str, None, None]:
    for char in response_message:
        time.sleep(0.02)
        yield char


def app() -> None:
    st.title("Hello Streamlit")

    human_message = st.chat_input()
    if not human_message:
        return

    with st.chat_message("human"):
        st.write(human_message)

    stream = generate_stream_response()
    with st.chat_message("ai"):
        st.write_stream(stream)


app()
