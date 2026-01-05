import streamlit as st
import weave
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.config import get_stream_writer
from langgraph.graph.state import CompiledStateGraph


def create_research_agent_tool(model_name: str, reasoning_effort: str) -> BaseTool:
    model = init_chat_model(
        model=model_name,
        model_provider="openai",
        reasoning_effort=reasoning_effort,
    )
    research_agent: CompiledStateGraph = create_agent(
        model=model,
        tools=[TavilySearch()],
        system_prompt="あなたは優秀なリサーチエージェントです。",
    )

    @tool
    def research_agent_tool(query: str) -> str:
        """リサーチエージェントを使用して情報を検索します"""
        stream_writer = get_stream_writer()

        stream_writer("サブエージェントの処理を開始します...")

        for chunk in research_agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="updates",
        ):
            for _, data in chunk.items():
                msgs = data["messages"]
                for msg in msgs:
                    stream_writer(msg)
                    filan_message = msg

        stream_writer("サブエージェントの処理を終了します")

        return filan_message.content

    return research_agent_tool


supervisor_system_prompt = """
あなたは優秀な監督者です。

以下のエージェントを使用して、ユーザーの質問に回答してください。
- リサーチエージェント
"""


def create_supervisor_agent(
    model_name: str, reasoning_effort: str
) -> CompiledStateGraph:
    model = init_chat_model(
        model=model_name,
        model_provider="openai",
        reasoning_effort=reasoning_effort,
    )

    tools = [
        create_research_agent_tool(model_name, reasoning_effort),
    ]

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=supervisor_system_prompt,
    )


def show_message(message: BaseMessage, ai_massage_type: str | None = None) -> None:
    if isinstance(message, HumanMessage):
        # ユーザーの入力の場合、そのまま表示する
        with st.chat_message(message.type):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        # マルチエージェントでAIのアイコンを変更するため、
        # ai_massage_typeが指定されている場合、それを使用する
        if ai_massage_type is not None:
            message_type = ai_massage_type
        else:
            message_type = message.type

        if len(message.tool_calls) == 0:
            # Function callingが選択されなかった場合、メッセージを表示する
            with st.chat_message(message_type):
                st.write(message.content)
        else:
            # Function callingが選択された場合、ツール名と引数を表示する
            for tool_call in message.tool_calls:
                with st.chat_message(message_type):
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
    weave.init("training-llm-app")

    st.title("Supervisor")

    with st.sidebar:
        model_name = st.selectbox(
            label="model_name",
            options=["gpt-5-nano", "gpt-5-mini", "gpt-5"],
            index=1,
        )
        reasoning_effort = st.selectbox(
            label="reasoning_effort",
            options=["minimal", "low", "medium", "high"],
            index=2,
        )

    # 会話履歴を初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    messages: list[BaseMessage] = st.session_state.messages

    # 会話履歴を表示
    for last_message in messages:
        show_message(last_message)

    # ユーザーの入力を受け付ける
    human_message = st.chat_input()
    if not human_message:
        return

    # ユーザーの入力を表示
    with st.chat_message("human"):
        st.write(human_message)

    # ユーザーの入力を会話履歴に追加
    messages.append(HumanMessage(content=human_message))

    # 応答を生成
    agent = create_supervisor_agent(
        model_name=model_name, reasoning_effort=reasoning_effort
    )

    # 新しいメッセージのみを追跡
    for stream_mode, chunk in agent.stream(
        {"messages": messages},
        stream_mode=["updates", "custom"],
    ):  # type: ignore[assignment]
        if stream_mode == "updates":
            assert isinstance(chunk, dict)
            for _, data in chunk.items():
                msgs = data["messages"]
                for msg in msgs:
                    messages.append(msg)
                    show_message(msg)

        elif stream_mode == "custom":
            if isinstance(chunk, BaseMessage):
                show_message(chunk, ai_massage_type="S")
            else:
                st.write(chunk)


app()
