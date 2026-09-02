from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import streamlit as st
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel


class ActionRequest(BaseModel):
    """人間のアクションを求めるときに画面に渡す情報を表すモデル"""

    name: str
    args: dict[str, Any]


class ActionRequests(BaseModel):
    action_requests: list[ActionRequest]


AgentStreamChunk = AIMessage | ToolMessage | ActionRequests


class MyAgent:
    def __init__(self) -> None:
        model = init_chat_model(
            model="gpt-5",
            model_provider="openai",
            reasoning_effort="medium",
        )
        self.agent = create_deep_agent(
            model=model,
            backend=FilesystemBackend(root_dir=".", virtual_mode=True),
            checkpointer=InMemorySaver(),
            interrupt_on={
                "write_file": {"allowed_decisions": ["approve", "edit", "reject"]},
                "edit_file": {"allowed_decisions": ["approve", "edit", "reject"]},
            },
        )

    def stream(self, message: str, thread_id: str) -> Iterator[AgentStreamChunk]:
        stream_input = {"messages": [HumanMessage(content=message)]}
        return self._stream(stream_input, thread_id)

    def _stream(self, stream_input: Any, thread_id: str) -> Iterator[AgentStreamChunk]:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        for chunk in self.agent.stream(
            input=stream_input,
            config=config,
        ):
            if "model" in chunk:
                messages = chunk["model"]["messages"]
                for m in messages:
                    yield m
            if "tools" in chunk:
                messages = chunk["tools"]["messages"]
                for m in messages:
                    yield m

        state = self.agent.get_state(config=config)
        if state.next:
            interrupts = state.tasks[0].interrupts[0].value
            interrupts_action_requests = interrupts["action_requests"]
            action_requests = [
                ActionRequest(
                    name=action_request["name"],
                    args=action_request["args"],
                )
                for action_request in interrupts_action_requests
            ]
            yield ActionRequests(action_requests=action_requests)

    def approve(self, thread_id: str) -> Iterator[AgentStreamChunk]:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        state = self.agent.get_state(config=config)
        interrupts = state.tasks[0].interrupts[0].value
        interrupts_action_requests = interrupts["action_requests"]
        interrupts_action_requests_count = len(interrupts_action_requests)

        decisions = [{"type": "approve"}] * interrupts_action_requests_count
        command: Command[tuple[()]] = Command(resume={"decisions": decisions})
        return self._stream(command, thread_id)

    def reject(self, feedback: str, thread_id: str) -> Iterator[AgentStreamChunk]:
        message = f"Rejected. Human feedback: {feedback}"
        decisions = [{"type": "reject", "message": message}]
        command: Command[tuple[()]] = Command(resume={"decisions": decisions})
        return self._stream(command, thread_id)

    def get_messages(self, thread_id: str) -> list[BaseMessage]:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        state_snapshot = self.agent.get_state(config=config)

        if "messages" in state_snapshot.values:
            return state_snapshot.values["messages"]  # type: ignore[no-any-return]
        else:
            return []

    def is_interrupted(self, thread_id: str) -> bool:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        state = self.agent.get_state(config=config)
        return bool(state.next)


class UIState:
    def __init__(self) -> None:
        self.agent = MyAgent()
        self.new_thread()
        self.show_approve_button = False

    def new_thread(self) -> None:
        self.thread_id = uuid4().hex


def show_message(message: BaseMessage) -> None:
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
        with st.chat_message(message.type):  # noqa: SIM117
            with st.expander(label="ツールの実行結果"):
                st.write(message.content)
    else:
        raise ValueError(
            f"Unknown message type. type: {type(message)}. message: {message}"
        )


def handle_agent_stream_chunk(chunk: AgentStreamChunk, ui_state: UIState) -> None:
    if type(chunk).__name__ == "ActionRequests":
        ui_state.show_approve_button = True
    else:
        show_message(chunk)


def app() -> None:
    load_dotenv(override=True)

    # UIStateを初期化
    if "deepagents_ui_state" not in st.session_state:
        st.session_state.deepagents_ui_state = UIState()
    ui_state: UIState = st.session_state.deepagents_ui_state

    with st.sidebar:
        # 新規スレッドボタン
        clicked = st.button("新規スレッド")
        if clicked:
            ui_state.new_thread()
            st.rerun()

    st.title("Deep Agents")
    st.write(f"thread_id: {ui_state.thread_id}")

    # 会話履歴を表示
    for m in ui_state.agent.get_messages(ui_state.thread_id):
        show_message(m)

    # 承認ボタンを表示
    if ui_state.show_approve_button:
        approved = st.button("承認")
        # 承認されたらエージェントを実行
        if approved:
            ui_state.show_approve_button = False
            with st.spinner():
                for chunk in ui_state.agent.approve(ui_state.thread_id):
                    handle_agent_stream_chunk(chunk, ui_state)
            # 会話履歴を表示するためrerun
            st.rerun()

    # ユーザーの指示を受け付ける
    human_input = st.chat_input()
    if human_input:
        show_message(HumanMessage(content=human_input))

        with st.spinner():
            if ui_state.show_approve_button:
                ui_state.show_approve_button = False
                for chunk in ui_state.agent.reject(human_input, ui_state.thread_id):
                    handle_agent_stream_chunk(chunk, ui_state)

            else:
                for chunk in ui_state.agent.stream(human_input, ui_state.thread_id):
                    handle_agent_stream_chunk(chunk, ui_state)

            # 会話履歴を表示するためrerun
            st.rerun()


app()
