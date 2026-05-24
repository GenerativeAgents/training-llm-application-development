from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.generate.types import GraphState, QualityJudgment
from app.llm import get_model

_SYSTEM_PROMPT_TEMPLATE = """\
あなたはカスタマーサポートの品質チェックAIです。
生成された回答メールの品質を「丁寧さ」の観点で評価してください。

- ビジネスメールとして適切な敬語が使われているか
- 失礼な表現や不適切な言い回しがないか
- OKまたはNGで判定し、判定理由を記述してください。"""

_USER_PROMPT_TEMPLATE = """\
以下の回答メールを評価してください。

<response_email>
<subject>{response_subject}</subject>
<body>
{response_body}
</body>
</response_email>"""


class QualityCheckResult(BaseModel):
    """品質チェック結果"""

    politeness: QualityJudgment = Field(description="丁寧さの判定")
    politeness_reason: str = Field(
        description="丁寧さの判定理由"
    )


async def quality_check(state: GraphState) -> GraphState:
    """生成された回答の品質チェックを行う。"""
    model = get_model(thinking=True)
    model_with_structure = model.with_structured_output(QualityCheckResult, method="json_schema")

    user_content = _USER_PROMPT_TEMPLATE.format(
        response_subject=state.get("response_subject", ""),
        response_body=state.get("response_body", ""),
    )
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT_TEMPLATE),
        HumanMessage(content=user_content),
    ]
    result: QualityCheckResult = await model_with_structure.ainvoke(messages)  # type: ignore[assignment]

    return {
        "politeness": result.politeness,
        "politeness_reason": result.politeness_reason,
    }
