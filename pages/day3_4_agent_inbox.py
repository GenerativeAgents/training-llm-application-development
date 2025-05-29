import sqlite3
from typing import Annotated, Any, Literal

import streamlit as st
from langchain_community.tools import ShellTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, Interrupt, RunnableConfig, interrupt
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

    # st.session_stateにagentを保存
    if "inbox_ui_state" not in st.session_state:
        st.session_state.inbox_ui_state = UIState()
    ui_state = st.session_state.inbox_ui_state

    # st.session_stateにagentを保存
    if "human_in_the_loop_agent" not in st.session_state:
        conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        checkpointer = SqliteSaver(conn=conn)
        st.session_state.human_in_the_loop_agent = Agent(checkpointer=checkpointer)
    agent = st.session_state.human_in_the_loop_agent

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
        st.write(f"thread_id: {thread_id}")

        # 会話履歴を表示
        messages = agent.get_messages(thread_id)
        show_messages(messages)

        # 次がhuman_review_nodeの場合は承認ボタンを表示
        if agent.is_next_human_review_node(thread_id):
            approved = st.button("承認")
            # 承認されたらエージェントを実行
            if approved:
                with st.spinner():
                    agent.handle_approve(thread_id)
                # 会話履歴を表示するためrerun
                st.rerun()

        # ユーザーの指示を受け付ける
        human_message = st.chat_input()
        if human_message:
            with st.spinner():
                agent.handle_human_message(human_message, thread_id)
                # 会話履歴を表示するためrerun
                st.rerun()


app()
