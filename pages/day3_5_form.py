import sqlite3
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel
from typing_extensions import TypedDict


class State(TypedDict):
    question: str
    draft_answer: str
    final_answer: str


class DraftAnswer(BaseModel):
    reasoning: str
    answer: str


def draft_node(state: State) -> dict:
    model = init_chat_model(
        model="gpt-5-nano",
        model_provider="openai",
        reasoning_effort="minimal",
    )
    model_with_structure = model.with_structured_output(DraftAnswer)

    messages = [
        SystemMessage(content="ユーザーの質問への解答案を作成してください。"),
        HumanMessage(content=state["question"]),
    ]
    output: DraftAnswer = model_with_structure.invoke(messages)  # type: ignore[assignment]

    return {"draft_answer": output.answer}


def human_review_node(state: State) -> dict:
    final_answer = interrupt(None)
    return {"final_answer": final_answer}


def send_final_answer_node(state: State) -> None:
    # 実際にはここでユーザーに回答を送信する
    print(f"最終回答: {state['final_answer']}")


def create_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(State)

    graph_builder.add_node("draft_node", draft_node)
    graph_builder.add_node("human_review_node", human_review_node)
    graph_builder.add_node("send_final_answer_node", send_final_answer_node)

    graph_builder.add_edge(START, "draft_node")
    graph_builder.add_edge("draft_node", "human_review_node")
    graph_builder.add_edge("human_review_node", "send_final_answer_node")
    graph_builder.add_edge("send_final_answer_node", END)

    conn = sqlite3.connect("tmp/checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)
    return graph_builder.compile(checkpointer=checkpointer)


def app() -> None:
    load_dotenv(override=True)

    st.title("お問い合わせフォーム")

    question = st.text_area(
        "お問い合わせ内容を入力してください",
        value="支給されているPCが壊れました。どうしたらいいですか？",
    )

    submit = st.button("送信")

    if not question or not submit:
        return

    # 回答を送信されたら、ワークフローを実行
    thread_id = uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    graph = create_graph()
    graph.invoke({"question": question}, config=config)

    st.info("お問い合わせを送信しました。")


if __name__ == "__main__":
    app()
