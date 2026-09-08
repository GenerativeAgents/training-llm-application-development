"""簡易的なコーディングエージェント（CLI）。

コマンド実行・ファイルの読み書きの 3 つのツールを LLM に与えて、
app/agent_loop.py のエージェントループで動かす。

    uv run python -m app.coding_agent [--work-dir DIR]

注意: コマンド実行やファイル書き込みを LLM に任せると、予期しない操作が行われる可能性があります。
講座では仕組みを理解するために使いますが、実際に使う際は動作する環境や権限に十分な注意が必要です。
"""

import argparse
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.agent_loop import Message, agent_loop

# エージェントが触れる範囲を限定するための作業ディレクトリ
WORK_DIR = Path("tmp/coding-agent").resolve()


# ---------- ツールの実装 ----------


def run_command(command: str) -> str:
    """作業ディレクトリでシェルコマンドを実行し、標準出力・標準エラー出力・終了コードを返す"""
    result = subprocess.run(
        command,
        shell=True,
        cwd=WORK_DIR,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.dumps(
        {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        },
        ensure_ascii=False,
    )


def _resolve_path(path: str) -> Path:
    """作業ディレクトリからの相対パスとして解決し、外に出ていたらエラーにする"""
    resolved = (WORK_DIR / path).resolve()
    if not resolved.is_relative_to(WORK_DIR):
        raise ValueError(f"作業ディレクトリの外は操作できません: {path}")
    return resolved


def read_file(path: str) -> str:
    """ファイルの内容を読んで返す"""
    return _resolve_path(path).read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    """ファイルに内容を書き込む（存在すれば上書き）"""
    target = _resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"{path} に {len(content)} 文字を書き込みました"


# ---------- ツールの定義（Chat Completions API に渡す tools） ----------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "作業ディレクトリでシェルコマンドを実行します。標準出力・標準エラー出力・終了コードをJSONで返します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "実行するシェルコマンド"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "作業ディレクトリ内のファイルの内容を読みます。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "作業ディレクトリからの相対パス"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "作業ディレクトリ内のファイルに内容を書き込みます。存在すれば上書きします。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "作業ディレクトリからの相対パス"},
                    "content": {"type": "string", "description": "書き込む内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS: dict[str, Callable[..., str]] = {
    "run_command": run_command,
    "read_file": read_file,
    "write_file": write_file,
}

SYSTEM_PROMPT = """あなたはコーディングエージェントです。
作業ディレクトリの中でファイルの作成・編集やコマンドの実行を行い、ユーザーの依頼を達成してください。
作業が終わったら、何をしたかを簡潔に報告してください。"""


# ---------- CLI ----------


def print_message(message: Message) -> None:
    """エージェントループから届いたメッセージをターミナルに表示する"""
    if message["role"] == "assistant":
        for tool_call in message.get("tool_calls") or []:
            function = tool_call["function"]
            print(f"[tool] {function['name']}({function['arguments']})")
        if message.get("content"):
            print(f"\n{message['content']}")
    elif message["role"] == "tool":
        content = str(message["content"])
        print(f"  -> {content[:200]}{'...' if len(content) > 200 else ''}")


def main() -> None:
    global WORK_DIR

    parser = argparse.ArgumentParser(description="簡易的なコーディングエージェント")
    parser.add_argument(
        "--work-dir",
        default=str(WORK_DIR),
        help="エージェントがファイル操作・コマンド実行を行うディレクトリ（既定: tmp/coding-agent）",
    )
    args = parser.parse_args()
    WORK_DIR = Path(args.work_dir).resolve()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    load_dotenv(override=True)

    messages: list[Message] = [{"role": "developer", "content": SYSTEM_PROMPT}]

    print(f"作業ディレクトリ: {WORK_DIR}")
    print("指示を入力してください（exit で終了）")

    while True:
        try:
            user_input = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.strip() in ("exit", "quit"):
            break
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})
        for message in agent_loop(messages, TOOLS, AVAILABLE_FUNCTIONS):
            print_message(message)


if __name__ == "__main__":
    main()
