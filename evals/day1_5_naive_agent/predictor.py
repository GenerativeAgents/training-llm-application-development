"""
Naive Agent 評価用 Predictor
"""

from typing import Any

import weave
from langchain_core.messages import HumanMessage
from weave import Model

from pages.day1_5_naive_agent import create_agent_with_tools


class NaiveAgentPredictor(Model):
    """
    Naive Agent の Predictor

    指定されたモデルで Naive Agent を実行し、結果を返す
    """

    model_name: str
    reasoning_effort: str = "medium"

    @weave.op
    def predict(self, message: str) -> dict[str, Any]:
        """
        メッセージを受け取り、エージェントを実行して結果を返す

        Args:
            message: ユーザーからの入力メッセージ

        Returns:
            messages: エージェントの実行結果（メッセージリスト）
        """
        agent = create_agent_with_tools(
            model_name=self.model_name,
            reasoning_effort=self.reasoning_effort,
        )

        result = agent.invoke({"messages": [HumanMessage(content=message)]})

        return {
            "messages": result["messages"],
        }
