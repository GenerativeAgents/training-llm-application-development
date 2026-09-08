import json

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from app.agent_loop import Message, Tool, agent_loop, function_to_tool
from app.coding_agent import run_command
from app.tools.web_search import web_search

# ---------- ツールの実装 ----------


def search_web(query: str) -> str:
    """最新の情報や知らないことを Web から検索し、本文の抜粋・URL・タイトル・公開日のリストを返す"""
    results = web_search(query, max_results=5)
    return json.dumps(results, ensure_ascii=False)


def turn_light(on: bool) -> str:
    """部屋の電気を ON/OFF する（True で ON、False で OFF）"""
    if on:
        return "LIGHT_ON"
    else:
        return "LIGHT_OFF"


# ---------- LLM に渡すツール（定義は関数の型ヒントと docstring から作られる） ----------

# 注意:
# 講座ではAIエージェントにできることを分かりやすく理解するためにコマンド実行ツールを使用します。
# しかし、コマンド実行ツールでは予期しないコマンドを実行される可能性があります。
# 実際に使用を検討する際は、AIエージェントが動作する環境などに十分な注意が必要です。
tools: list[Tool] = [
    function_to_tool(search_web),
    function_to_tool(run_command),
    function_to_tool(turn_light),
]

system_prompt = """
ファイルの作成を依頼された場合、run_commandでechoコマンドを使用してください。
"""


def show_message(message: Message) -> None:
    if message["role"] == "user":
        # ユーザーの入力の場合、そのまま表示する
        with st.chat_message("human"):
            st.write(message["content"])
    elif message["role"] == "assistant":
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) == 0:
            # Function callingが選択されなかった場合、メッセージを表示する
            with st.chat_message("ai"):
                st.write(message["content"])
        else:
            # Function callingが選択された場合、ツール名と引数を表示する
            for tool_call in tool_calls:
                function = tool_call["function"]
                with st.chat_message("ai"):
                    st.write(
                        f"'{function['name']}' を {function['arguments']} で実行します",
                    )
    elif message["role"] == "tool":
        # ツールの実行結果を折りたたんで表示する
        with st.chat_message("tool"):
            with st.expander(label="ツールの実行結果"):
                st.write(message["content"])


def app() -> None:
    load_dotenv(override=True)

    st.title("Naive Agent")

    # 電灯の状態を初期化
    if "is_light_on" not in st.session_state:
        st.session_state.is_light_on = False

    # 会話履歴を初期化（システムプロンプトは会話履歴とは別に先頭へ付ける）
    if "messages" not in st.session_state:
        st.session_state.messages = []
    messages: list[Message] = st.session_state.messages

    # 会話履歴を表示
    for message in messages:
        show_message(message)

    # ユーザーの入力を受け付ける
    human_message = st.chat_input()

    # 入力があった場合、エージェントを実行
    if human_message:
        # ユーザーの入力を表示して会話履歴に追加
        user_message: Message = {"role": "user", "content": human_message}
        show_message(user_message)
        messages.append(user_message)

        # エージェントループを回し、届いたメッセージを順に表示する
        all_messages = [{"role": "developer", "content": system_prompt}, *messages]
        for message in agent_loop(all_messages, tools):
            show_message(message)
            messages.append(message)

            if message["role"] == "tool":
                if message["content"] == "LIGHT_ON":
                    st.session_state.is_light_on = True
                elif message["content"] == "LIGHT_OFF":
                    st.session_state.is_light_on = False

    # サイドバーに電灯の画像を表示
    with st.sidebar:
        if st.session_state.is_light_on:
            light_on_off = "on"
        else:
            light_on_off = "off"

        st.image(Image.open(f"data/light-{light_on_off}.png"))


app()
