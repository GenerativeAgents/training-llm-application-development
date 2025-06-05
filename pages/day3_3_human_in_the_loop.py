import sqlite3
from typing import Annotated, Any, Literal
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools import ShellTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, RunnableConfig, interrupt
from pydantic import BaseModel
from typing_extensions import TypedDict


class HumanReviewApprove(BaseModel):
    pass


class HumanReviewFeedback(BaseModel):
    feedback: str


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class Agent:
    def __init__(self, checkpointer: BaseCheckpointSaver) -> None:
        self.llm = ChatOpenAI(model="gpt-4.1-nano")
        self.tools = [TavilySearchResults(), ShellTool()]

        graph_builder = StateGraph(State)

        graph_builder.add_node("llm_node", self._llm_node)
        graph_builder.add_node("tool_node", ToolNode(self.tools))
        graph_builder.add_node("human_review_node", self._human_review_node)

        graph_builder.add_edge(START, "llm_node")
        graph_builder.add_conditional_edges(
            "llm_node",
            self._is_tool_use,
            {
                True: "human_review_node",
                False: END,
            },
        )
        graph_builder.add_edge("tool_node", "llm_node")

        self.graph = graph_builder.compile(
            checkpointer=checkpointer,
        )

    def _llm_node(self, state: State) -> dict[str, Any]:
        llm_with_tools = self.llm.bind_tools(self.tools)
        ai_message = llm_with_tools.invoke(state["messages"])
        return {"messages": [ai_message]}

    def _human_review_node(
        self, state: dict, config: RunnableConfig
    ) -> Command[Literal["tool_node", "llm_node"]]:
        human_review = interrupt(None)

        if isinstance(human_review, HumanReviewApprove):
            return Command(goto="tool_node")
        elif isinstance(human_review, HumanReviewFeedback):
            # ツールの呼び出しが失敗したことをStateに追加
            last_message = state["messages"][-1]
            tool_reject_message = ToolMessage(
                content="Tool call rejected",
                status="error",
                name=last_message.tool_calls[0]["name"],
                tool_call_id=last_message.tool_calls[0]["id"],
            )
            return Command(
                goto="llm_node",
                update={"messages": [tool_reject_message, human_review.feedback]},
            )
        else:
            raise ValueError(f"Unknown human review: {human_review}")

    def _is_tool_use(self, state: State) -> bool:
        last_message = state["messages"][-1]
        return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0

    def handle_human_message(self, human_message: str, thread_id: str) -> None:
        config = {"configurable": {"thread_id": thread_id}}

        if self.is_next_human_review_node(thread_id):
            self.graph.invoke(
                Command(resume=HumanReviewFeedback(feedback=human_message)),
                config=config,
            )
        else:
            self.graph.invoke(
                input={"messages": [HumanMessage(content=human_message)]},
                config=config,
            )

    def get_messages(self, thread_id: str) -> list[BaseMessage]:
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = self.graph.get_state(config=config)

        if "messages" in state_snapshot.values:
            return state_snapshot.values["messages"]
        else:
            return []

    def handle_approve(self, thread_id: str) -> None:
        config = {"configurable": {"thread_id": thread_id}}
        self.graph.invoke(Command(resume=HumanReviewApprove()), config=config)

    def is_next_human_review_node(self, thread_id: str) -> bool:
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = self.graph.get_state(config=config)
        graph_next = state_snapshot.next
        return len(graph_next) != 0 and graph_next[0] == "human_review_node"

    def mermaid_png(self) -> bytes:
        return self.graph.get_graph().draw_mermaid_png()


def show_messages(messages: list[BaseMessage]) -> None:
    for message in messages:
        if isinstance(message, HumanMessage):
            # ユーザーの入力の場合、そのまま表示する
            with st.chat_message(message.type):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            if len(message.tool_calls) == 0:
                # Function callingが選択されなかった場合、メッセージを表示する
                with st.chat_message(message.type):
                    st.write(message.content)
            else:
                # Function callingが選択された場合、ツール名と引数を表示する
                for tool_call in message.tool_calls:
                    with st.chat_message(message.type):
                        st.write(
                            f"'{tool_call['name']}' を {tool_call['args']} で実行します",
                        )
        elif isinstance(message, ToolMessage):
            # ツールの実行結果を折りたたんで表示する
            with st.chat_message(message.type):
                with st.expander(label="ツールの実行結果"):
                    st.write(message.content)
        else:
            raise ValueError(f"Unknown message type: {message}")


class UIState:
    def __init__(self, agent: Agent, thread_id: str | None = None) -> None:
        self.agent: Agent = agent
        self.new_thread()

    def new_thread(self) -> None:
        self.thread_id = uuid4().hex


def app(thread_id: str | None = None) -> None:
    load_dotenv(override=True)

    # st.session_stateにagentを保存
    if "human_in_the_loop_ui_state" not in st.session_state:
        conn = sqlite3.connect("tmp/checkpoints.sqlite", check_same_thread=False)
        checkpointer = SqliteSaver(conn=conn)
        st.session_state.human_in_the_loop_ui_state = UIState(
            agent=Agent(checkpointer=checkpointer),
            thread_id=thread_id,
        )
    ui_state = st.session_state.human_in_the_loop_ui_state
    if thread_id is not None:
        ui_state.thread_id = thread_id

    with st.sidebar:
        # 新規スレッドボタン
        clicked = st.button("新規スレッド")
        if clicked:
            ui_state.new_thread()
            st.rerun()

        # グラフを表示
        st.image(ui_state.agent.mermaid_png())

    st.title("Agent")
    st.write(f"thread_id: {ui_state.thread_id}")

    # 会話履歴を表示
    messages = ui_state.agent.get_messages(ui_state.thread_id)
    show_messages(messages)

    # 次がhuman_review_nodeの場合は承認ボタンを表示
    if ui_state.agent.is_next_human_review_node(ui_state.thread_id):
        approved = st.button("承認")
        # 承認されたらエージェントを実行
        if approved:
            with st.spinner():
                ui_state.agent.handle_approve(ui_state.thread_id)
            # 会話履歴を表示するためrerun
            st.rerun()

    # ユーザーの指示を受け付ける
    human_message = st.chat_input()
    if human_message:
        with st.spinner():
            ui_state.agent.handle_human_message(human_message, ui_state.thread_id)
            # 会話履歴を表示するためrerun
            st.rerun()


if __name__ == "__main__":
    app()
