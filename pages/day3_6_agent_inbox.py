import sqlite3

import streamlit as st
import weave
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, Interrupt, RunnableConfig
from pydantic import BaseModel

from pages.day3_5_form import create_graph


class UIState(BaseModel):
    selected_thread_id: str | None = None


class InterruptThread(BaseModel):
    thread_id: str
    question: str
    draft_answer: str
    interrupts: list[Interrupt]


def get_interrupt_threads(checkpointer: SqliteSaver) -> list[InterruptThread]:
    """
    この関数では、LangGraphのCheckpointerからInterrupt状態のスレッド一覧を取得します。

    注意:
    この関数の実装はLangGraphのCheckpointerの構造に強く依存しています。
    実際にはCheckpointerの構造に強く依存するコードを使うことはおすすめしません。
    LangGraphでInterrupt状態のスレッド一覧を取得したい場合は、
    LangGraph ServerのAPIを使用するか、自前でスレッドの状態を管理することをおすすめします。
    """
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
        last_checkpoint_state = last_checkpoint_tuple.checkpoint["channel_values"]
        question = last_checkpoint_state["question"]
        draft_answer = last_checkpoint_state["draft_answer"]
        pending_writes = last_checkpoint_tuple.pending_writes

        if pending_writes and pending_writes[0][1] == "__interrupt__":
            interrupts = pending_writes[0][2]
            interrupt_thread = InterruptThread(
                thread_id=thread_id,
                question=question,
                draft_answer=draft_answer,
                interrupts=interrupts,
            )
            interrupt_threads.append(interrupt_thread)

    return interrupt_threads


def app() -> None:
    load_dotenv(override=True)
    weave.init("training-llm-app")

    st.title("Inbox")

    conn = sqlite3.connect("tmp/checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)
    interrupt_threads = get_interrupt_threads(checkpointer)

    # stateを初期化
    if "agent_inbox_ui_state" not in st.session_state:
        st.session_state.agent_inbox_ui_state = UIState()
    ui_state = st.session_state.agent_inbox_ui_state

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("人間への確認依頼一覧")
        for interrupt_thread in interrupt_threads:
            with st.container(border=True):
                st.write(interrupt_thread.question)
                clicked = st.button("対応する", key=interrupt_thread.thread_id)
                if clicked:
                    ui_state.selected_thread_id = interrupt_thread.thread_id

    with col2:
        if ui_state.selected_thread_id is None:
            st.info("確認依頼を選択してください")
            return
        else:
            st.subheader("回答調整欄")

            intrrupt_thread: InterruptThread = next(
                filter(
                    lambda t: t.thread_id == ui_state.selected_thread_id,
                    interrupt_threads,
                )
            )
            st.write("### 質問")
            st.write(intrrupt_thread.question)
            st.write("### 回答ドラフト")
            final_answer = st.text_area(
                "回答", value=intrrupt_thread.draft_answer, height=400
            )

            submit = st.button("回答送信")

            if not final_answer or not submit:
                return

            thread_id = intrrupt_thread.thread_id
            config = {"configurable": {"thread_id": thread_id}}
            graph = create_graph()
            graph.invoke(Command(resume=final_answer), config=config)

            st.info("回答を送信しました。")

            ui_state.selected_thread_id = None
            st.rerun()


app()
