from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.generate.types import GraphState
from app.llm import get_model

_SYSTEM_PROMPT_TEMPLATE = """\
あなたは株式会社サンプルエージェントのカスタマーサポート担当です。
お客様からのお問い合わせに対して、丁寧なビジネスメール形式で回答を作成してください。

<rules>
- 敬語を使用すること
- 宛名（会社名・お客様名）を含めること
- 挨拶文から始めること
- 具体的で役立つ回答を提供すること
- 締めの挨拶で終わること
- 回答件名はお問い合わせ内容に基づいた適切な件名にすること
</rules>"""

_USER_PROMPT_TEMPLATE = """\
以下のお問い合わせに対して回答メールを作成してください。件名と本文を分けて出力してください。

<inquiry>
<topic>{topic}</topic>
<customer_name>{customer_name}</customer_name>
<company_name>{company_name}</company_name>
<content>
{content}
</content>
</inquiry>"""


class GeneratedResponse(BaseModel):
    """生成された回答メール"""

    response_subject: str = Field(
        description="回答メールの件名（お問い合わせ内容から適切な件名を生成）"
    )
    response_body: str = Field(description="回答メールの本文")


async def generate_response(state: GraphState) -> GraphState:
    """問い合わせ内容に基づいて回答メールを生成する。"""
    model = get_model(thinking=True)
    model_with_structure = model.with_structured_output(GeneratedResponse, method="json_schema")

    user_content = _USER_PROMPT_TEMPLATE.format(
        topic=state.get("topic", "other"),
        customer_name=state["customer_name"],
        company_name=state.get("company_name") or "（なし）",
        content=state["content"],
    )
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT_TEMPLATE),
        HumanMessage(content=user_content),
    ]
    result: GeneratedResponse = await model_with_structure.ainvoke(messages)  # type: ignore[assignment]

    return {
        "response_subject": result.response_subject,
        "response_body": result.response_body,
    }
