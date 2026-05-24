from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.generate.types import GraphState, TopicType
from app.llm import get_model

_SYSTEM_PROMPT_TEMPLATE = """\
あなたはカスタマーサポートのトピック分類AIです。
お客様からのお問い合わせ内容を分析し、以下の4つのカテゴリに分類してください。

- development: AIエージェント開発支援サービスに関する質問（カスタム開発、受託開発、見積もり、費用、開発期間など）
- product: AgentBoardプロダクトに関する質問（機能、料金、プラン、導入、連携、API、仕様、セキュリティなど）
- other: 上記に当てはまらない一般的な問い合わせ
- spam: スパム、営業メール、広告、SEO対策の提案など

confidenceは分類の確信度を0.0〜1.0で表してください。"""

_USER_PROMPT_TEMPLATE = """\
{content}"""


class TopicClassification(BaseModel):
    """トピック分類結果"""

    topic: TopicType = Field(description="分類されたトピック")
    confidence: float = Field(description="分類の確信度 (0.0-1.0)", ge=0.0, le=1.0)


async def classify_topic(state: GraphState) -> GraphState:
    """問い合わせ内容をトピック分類する。"""
    model = get_model()
    model_with_structure = model.with_structured_output(TopicClassification, method="json_schema")

    user_content = _USER_PROMPT_TEMPLATE.format(content=state["content"])
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT_TEMPLATE),
        HumanMessage(content=user_content),
    ]
    result: TopicClassification = await model_with_structure.ainvoke(messages)  # type: ignore[assignment]

    return {
        "topic": result.topic,
        "classification_confidence": result.confidence,
    }
