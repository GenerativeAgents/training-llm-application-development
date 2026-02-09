"""
Naive Agent 評価用スコアラー

Trajectory評価：エージェントの実行履歴を分析し、
期待されるツールが呼び出されたかを評価する
"""

from typing import Any

import weave
from langchain_core.messages import AIMessage


class ToolCallScorer(weave.Scorer):
    """
    エージェントが期待されるツールを呼び出したかを評価

    Trajectory評価として、エージェントの実行履歴（メッセージリスト）を分析し、
    AIMessageのtool_callsから期待されるツールが呼ばれたかをチェックする
    """

    @weave.op
    async def score(self, output: dict[str, Any], expected_tool: str) -> dict:
        """
        Args:
            output: Predictorの出力（messages キーを含む）
            expected_tool: 呼び出されるべきツール名

        Returns:
            tool_called: 期待するツールが呼ばれたか（1 または 0）
            called_tools: 実際に呼ばれたツール一覧（デバッグ用）
        """
        messages = output.get("messages", [])

        called_tools = []

        for message in messages:
            if isinstance(message, AIMessage):
                if len(message.tool_calls) > 0:
                    for tool_call in message.tool_calls:
                        tool_name = tool_call["name"]
                        if tool_name not in called_tools:
                            called_tools.append(tool_name)

        tool_called = 1 if expected_tool in called_tools else 0

        return {
            "tool_called": tool_called,
            "called_tools": called_tools,
        }
