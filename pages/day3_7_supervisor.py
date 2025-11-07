import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
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
        final_state = research_agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        return final_state["messages"][-1].content

    return research_agent_tool


def create_report_agent_tool(model_name: str, reasoning_effort: str) -> BaseTool:
    model = init_chat_model(
        model=model_name,
        model_provider="openai",
        reasoning_effort=reasoning_effort,
    )
    report_agent: CompiledStateGraph = create_agent(
        model=model,
        tools=[TavilySearch()],
        system_prompt="あなたは優秀なレポートエージェントです。",
    )

    @tool
    def report_agent_tool(query: str) -> str:
        """レポートエージェントを使用してレポートを作成します"""
        final_state = report_agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        return final_state["messages"][-1].content

    return report_agent_tool


supervisor_system_prompt = """
あなたは優秀な監督者です。

以下のエージェントを使用して、ユーザーの質問に回答してください。
- リサーチエージェント
- レポートエージェント
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
        create_report_agent_tool(model_name, reasoning_effort),
    ]

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=supervisor_system_prompt,
    )


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
        with st.chat_message(message.type):
            with st.expander(label="ツールの実行結果"):
                st.write(message.content)
    else:
        raise ValueError(f"Unknown message type: {message}")


def app() -> None:
    load_dotenv(override=True)

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
    new_messages = []
    for s in agent.stream({"messages": messages}, stream_mode="values"):  # type: ignore[assignment]
        # ストリームから全メッセージを取得
        all_messages = s["messages"]

        # 既存のメッセージ数以降の新しいメッセージのみを処理
        for msg in all_messages[len(messages) :]:
            if msg not in new_messages:
                new_messages.append(msg)
                show_message(msg)

    # 新しいメッセージのみを履歴に追加
    messages.extend(new_messages)


app()
