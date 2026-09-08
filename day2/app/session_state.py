"""Streamlit の session_state をページごとに切り分けるためのヘルパー。

st.session_state はページ間で共有されるため、別のページが同じキー（例: messages）に
違う形式の値を入れていると、ページを切り替えたときにそれを読んで壊れる。
各ページの app() の先頭で呼び、前回と別のページなら session_state を空にする。
"""

from pathlib import Path

import streamlit as st


def reset_session_state_on_page_change(page_file: str) -> None:
    """前回実行したページと違うページなら st.session_state をリセットする（page_file には __file__ を渡す）"""
    page = Path(page_file).stem
    if st.session_state.get("current_page") != page:
        st.session_state.clear()
        st.session_state.current_page = page
