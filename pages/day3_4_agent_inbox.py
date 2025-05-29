import sqlite3

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Interrupt, RunnableConfig
from pydantic import BaseModel


class UIState(BaseModel):
    selected_thread_id: str | None = None


class InterruptThread(BaseModel):
    thread_id: str
    thread_title: str
    interrupts: list[Interrupt]


def app() -> None:
    st.title("Toy Agent Inbox")

    conn = sqlite3.connect("tmp/checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)

    checkpoints = checkpointer.list(config=None)

    thread_ids = set()
    for checkpoint in checkpoints:
        thread_id = checkpoint.config["configurable"]["thread_id"]
        thread_ids.add(thread_id)

    interrupt_threads: list[InterruptThread] = []
    for thread_id in thread_ids:
        config = RunnableConfig(configurable={"thread_id": thread_id})
        checkpoint_tuples = checkpointer.list(config)
        last_checkpoint_tuple = next(checkpoint_tuples)
        first_human_message = last_checkpoint_tuple.checkpoint["channel_values"][
            "messages"
        ][0].content
        pending_writes = last_checkpoint_tuple.pending_writes

        if pending_writes and pending_writes[0][1] == "__interrupt__":
            interrupts = pending_writes[0][2]
            interrupt_thread = InterruptThread(
                thread_id=thread_id,
                thread_title=first_human_message,
                interrupts=interrupts,
            )
            interrupt_threads.append(interrupt_thread)

    # stateを初期化
    if "inbox_ui_state" not in st.session_state:
        st.session_state.inbox_ui_state = UIState()
    ui_state = st.session_state.inbox_ui_state

    col1, col2 = st.columns(2)

    with col1:
        for interrupt_thread in interrupt_threads:
            with st.container(border=True):
                st.subheader(interrupt_thread.thread_title)
                st.write(f"thread_id: {interrupt_thread.thread_id}")
                clicked = st.button("詳細", key=interrupt_thread.thread_id)
                if clicked:
                    ui_state.selected_thread_id = interrupt_thread.thread_id

    with col2:
        # スレッドが選択されていない場合は何も表示しない
        if ui_state.selected_thread_id is None:
            return

        thread_id = ui_state.selected_thread_id
        st.session_state.thread_id = thread_id

        from pages.day3_3_human_in_the_loop import app as hil_app

        hil_app()


app()
