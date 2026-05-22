from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv(override=True)


class State(TypedDict):
    # 入力
    inquiry: str
    # スパム判定の出力
    is_spam: bool
    reason_in_japanese: str
    # 返信文生成の出力
    response: str | None


model = init_chat_model(
    model="gpt-5-mini",
    model_provider="openai",
    reasoning_effort="medium",
)


class SpamJudgeResult(BaseModel):
    is_spam: bool = Field(description="スパムかどうか")
    reason_in_japanese: str = Field(description="判定理由")


def spam_judge_node(state: State) -> dict[str, Any]:
    model_with_structure = model.with_structured_output(SpamJudgeResult)
    prompt = [
        SystemMessage(content="以下のお問い合わせがスパムか判定してください。"),
        HumanMessage(content=f"お問い合わせ: {state['inquiry']}"),
    ]
    return model_with_structure.invoke(prompt)


def generate_answer_node(state: State) -> dict[str, Any]:
    prompt = [
        SystemMessage(content="以下のお問い合わせに対する、返信文を出力してください。"),
        HumanMessage(content=f"お問い合わせ: {state['inquiry']}"),
    ]
    ai_message = model.invoke(prompt)
    return {"response": ai_message.content}


def create_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(State)

    graph_builder.add_node("spam_judge_node", spam_judge_node)
    graph_builder.add_node("generate_answer_node", generate_answer_node)

    graph_builder.add_edge(START, "spam_judge_node")

    # checkノードから次のノードへの遷移に条件付きエッジを定義
    # state.current_judgeの値がTrueならENDノードへ、Falseならselectionノードへ
    graph_builder.add_conditional_edges(
        "spam_judge_node",
        lambda state: state["is_spam"],
        {True: END, False: "generate_answer_node"},
    )

    graph_builder.add_edge("generate_answer_node", END)

    return graph_builder.compile()


def app() -> None:

    st.title("お問い合わせ返信文生成")

    if "graph" not in st.session_state:
        st.session_state.graph = create_graph()
    graph = st.session_state.graph

    inquiry = st.text_area(
        label="お問い合わせ内容",
        value="AIエージェントの開発を依頼したいです。",
    )

    clicked = st.button("実行", disabled=not inquiry)
    if not clicked:
        return

    with st.spinner("実行中..."):
        initial_state = {"inquiry": inquiry}
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            keys = chunk.keys()
            for key in keys:
                state = chunk[key]

                if key == "spam_judge_node":
                    is_spam = state["is_spam"]
                    reason_in_japanese = state["reason_in_japanese"]
                    with st.expander(f"スパム判定結果: {is_spam}"):
                        st.subheader("判定理由")
                        st.markdown(reason_in_japanese)
                elif key == "generate_answer_node":
                    response = state["response"]
                    st.subheader("返信文")
                    st.markdown(response)


app()
