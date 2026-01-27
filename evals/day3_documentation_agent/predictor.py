"""
Documentation Agent 評価用 Predictor
"""

from typing import Any

import weave
from langchain_openai import ChatOpenAI
from weave import Model

from app.documentation_agent.agent import DocumentationAgent


class DocumentationAgentPredictor(Model):
    """
    Documentation Agent の Predictor

    指定されたモデルで DocumentationAgent を実行し、要件定義書を生成する
    """

    model_name: str
    k: int = 5  # ペルソナ生成数

    @weave.op
    def predict(self, user_request: str) -> dict[str, Any]:
        """
        ユーザーリクエストから要件定義書を生成

        Args:
            user_request: アプリケーション開発要求

        Returns:
            requirements_doc: 生成された要件定義書
        """
        llm = ChatOpenAI(model=self.model_name, temperature=0.0)
        agent = DocumentationAgent(llm=llm, k=self.k)

        requirements_doc = agent.run(user_request=user_request)

        return {
            "requirements_doc": requirements_doc,
        }
