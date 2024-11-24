from typing import Annotated, Any
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class Agent:
    def __init__(self, checkpointer: BaseCheckpointSaver) -> None:
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.tools = [TavilySearchResults()]

        graph_builder = StateGraph(State)
        graph_builder.add_node("llm_node", self._llm_node)
        graph_builder.add_node("tool_node", ToolNode(self.tools))
        graph_builder.add_node("human_review_node", self._human_review_node)
        graph_builder.add_conditional_edges(
            "llm_node",
            self._is_tool_use,
            {
                True: "human_review_node",
                False: END,
            },
        )
        graph_builder.add_conditional_edges(
            "human_review_node",
            self._is_human_approved,
            {
                True: "tool_node",
                False: "llm_node",
            },
        )
        graph_builder.add_edge("tool_node", "llm_node")
        graph_builder.add_edge(START, "llm_node")
        self.graph = graph_builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["human_review_node"],
        )

    def _llm_node(self, state: State) -> dict[str, Any]:
        llm_with_tools = self.llm.bind_tools(self.tools)
        ai_message = llm_with_tools.invoke(state["messages"])
        return {"messages": [ai_message]}

    def _human_review_node(self, state: dict) -> None:
        pass

    def _is_tool_use(self, state: State) -> bool:
        last_message = state["messages"][-1]
        return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0

    def _is_human_approved(self, state: State) -> bool:
        # 最後のメッセージがHumanMessageの場合は承認されずに追加の入力があったということなので、
        # 最後のメッセージがAIMessageの場合は承認されたとみなすことができる
        last_message = state["messages"][-1]
        return isinstance(last_message, AIMessage)

    def handle_human_message(self, human_message: str, thread_id: str) -> None:
        # 承認待ちの状態でhuman_messageが送信されるのは、ツールの呼び出しを修正したい状況のため、
        # 次がhuman_review_nodeの場合、ツールの呼び出しが失敗したことをStateに追加
        # 参考: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/review-tool-calls/#give-feedback-to-a-tool-call
        if self.is_next_human_review_node(thread_id):
            last_message = self.get_messages(thread_id)[-1]
            tool_reject_message = ToolMessage(
                content="Tool call rejected",
                status="error",
                name=last_message.tool_calls[0]["name"],
                tool_call_id=last_message.tool_calls[0]["id"],
            )
            config = {"configurable": {"thread_id": thread_id}}
            self.graph.update_state(
                config=config,
                values={"messages": [tool_reject_message]},
                as_node="human_review_node",
            )

        config = {"configurable": {"thread_id": thread_id}}
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
        self.graph.invoke(input=None, config=config)

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


def app() -> None:
    load_dotenv(override=True)

    st.title("Agent")

    # st.session_stateにagentを保存
    if "agent" not in st.session_state:
        checkpointer = MemorySaver()
        st.session_state.agent = Agent(checkpointer=checkpointer)
    agent = st.session_state.agent

    # グラフを表示
    with st.sidebar:
        st.image(agent.mermaid_png())

    # st.session_stateにthread_idを保存
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid4().hex
    thread_id = st.session_state.thread_id
    st.write(f"thread_id: {thread_id}")

    # ユーザーの指示を受け付ける
    human_message = st.chat_input()
    if human_message:
        with st.spinner():
            agent.handle_human_message(human_message, thread_id)

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


app()
