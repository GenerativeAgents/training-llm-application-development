import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.tools import ShellTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph.state import CompiledStateGraph
from PIL import Image


@tool
def turn_light(on: bool) -> str:
    """部屋の電気をON/OFFするツールです"""
    if on:
        return "LIGHT_ON"
    else:
        return "LIGHT_OFF"


system_prompt = """
ファイルの作成を依頼された場合、terminalでechoコマンドを使用してください。
"""


def create_agent_with_tools(
    model_name: str, reasoning_effort: str
) -> CompiledStateGraph:
    tools = [
        TavilySearch(max_results=5),
        # 注意:
        # 講座ではAIエージェントにできることを分かりやすく理解するためにShellToolを使用します。
        # しかし、ShellToolでは予期しないコマンドを実行される可能性があります。
        # 実際にShellToolの使用を検討する際は、AIエージェントが動作する環境などに十分な注意が必要です。
        ShellTool(),
        turn_light,
    ]

    model = init_chat_model(
        model=model_name,
        model_provider="openai",
        reasoning_effort=reasoning_effort,
    )
    return create_agent(model=model, tools=tools, system_prompt=system_prompt)


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

    st.title("Naive Agent")

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

    # 電灯の状態を初期化
    if "is_light_on" not in st.session_state:
        st.session_state.is_light_on = False

    # 会話履歴を初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    messages: list[BaseMessage] = st.session_state.messages

    # 会話履歴を表示
    for last_message in messages:
        show_message(last_message)

    # ユーザーの入力を受け付ける
    human_message = st.chat_input()

    # 入力があった場合、エージェントを実行
    if human_message:
        # ユーザーの入力を表示
        with st.chat_message("human"):
            st.write(human_message)

        # ユーザーの入力を会話履歴に追加
        messages.append(HumanMessage(content=human_message))

        # 応答を生成
        agent = create_agent_with_tools(
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

                if isinstance(msg, ToolMessage):
                    if msg.content == "LIGHT_ON":
                        st.session_state.is_light_on = True
                    elif msg.content == "LIGHT_OFF":
                        st.session_state.is_light_on = False

        # 新しいメッセージのみを履歴に追加
        messages.extend(new_messages)

    # サイドバーに電灯の画像を表示
    with st.sidebar:
        if st.session_state.is_light_on:
            light_on_off = "on"
        else:
            light_on_off = "off"

        st.image(Image.open(f"data/light-{light_on_off}.png"))


app()
