import streamlit as st
import weave
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langgraph.graph.state import CompiledStateGraph

# from langchain_community.tools import ShellTool
# from langchain_core.tools import tool

# @tool
# def turn_light(on: bool) -> str:
#     """部屋の電気をON/OFFするツールです"""
#     if on:
#         return "電気をつけました"
#     else:
#         return "電気を消しました"


def create_agent_with_tools(
    model_name: str, reasoning_effort: str
) -> CompiledStateGraph:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory="./tmp/chroma",
    )
    retriever = vector_store.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="langsmith-retriever",
        description="LangSmithの最新ドキュメントを検索します。LangSmithについて質問された場合に使用してください。",
    )
    # 注意:
    # 講座ではAIエージェントにできることを分かりやすく理解するためにShellToolを使用します。
    # しかし、ShellToolでは予期しないコマンドを実行される可能性があります。
    # 実際にShellToolの使用を検討する際は、AIエージェントが動作する環境などに十分な注意が必要です。
    tools = [
        retriever_tool,
        # ShellTool(),
        # turn_light,
    ]

    model = init_chat_model(
        model=model_name,
        model_provider="openai",
        reasoning_effort=reasoning_effort,
    )
    return create_agent(model=model, tools=tools)


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
    weave.init("training-llm-app")

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

    # 新しいメッセージのみを履歴に追加
    messages.extend(new_messages)


app()
