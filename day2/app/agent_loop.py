"""Chat Completions API の Function calling でエージェントループを回す。

part1_2 のノートブックで実装した agent_loop と同じ構造で、表示に使えるように
追加されたメッセージ（LLM の応答・ツールの実行結果）を順に yield する。
CLI（app/coding_agent.py）では print し、Streamlit（pages/part1_3_agent.py）では画面に描く。
"""

import json
from collections.abc import Callable, Iterator
from typing import Any

from openai import OpenAI

Message = dict[str, Any]


def agent_loop(
    messages: list[Message],
    tools: list[dict[str, Any]],
    available_functions: dict[str, Callable[..., str]],
    *,
    model: str = "gpt-5.6-luna",
    max_iterations: int = 20,
) -> Iterator[Message]:
    """LLM がツールを使いたいと応答する限り、ツールを実行して結果を渡し続ける。

    messages は呼び出し側のリストをそのまま更新する（会話履歴として使い回せる）。
    """
    client = OpenAI()

    for _ in range(max_iterations):
        response = client.chat.completions.create(  # type: ignore[call-overload]
            model=model,
            messages=messages,
            tools=tools,
            # Chat Completions API で Function tools を使う場合、reasoning_effort は "none" のみ対応
            reasoning_effort="none",
        )
        response_message = response.choices[0].message.to_dict()
        messages.append(response_message)
        yield response_message

        # ツールを使わない応答なら、それが最終的な回答
        tool_calls = response_message.get("tool_calls")
        if not tool_calls:
            return

        # ツールを使いたいという応答なら、ツールを実行して結果を messages に追加し、再度 LLM を呼び出す
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            function_response = available_functions[function_name](**function_args)
            tool_message: Message = {
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
            messages.append(tool_message)
            yield tool_message

    limit_message: Message = {
        "role": "assistant",
        "content": "（反復回数の上限に達しました）",
    }
    messages.append(limit_message)
    yield limit_message
