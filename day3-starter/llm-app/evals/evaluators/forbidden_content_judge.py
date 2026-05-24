from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm import get_model

_SYSTEM_PROMPT_TEMPLATE = """\
あなたはカスタマーサポートの回答メールに禁止コンテンツが含まれていないかを\
チェックする品質管理の専門家です。

<forbidden_items>
以下のいずれかが回答メールに含まれている場合、「fail」と判定してください。

1. 見積り金額の提示: 具体的な金額、料金、費用の数値を提示している場合。\
「お見積もりを作成します」のような案内は問題ありませんが、\
「月額10万円」「開発費用は500万円」のように具体的な数値を提示している場合はNGです。

2. 未確定情報の断定: 確認が必要な事項を断定的に述べている場合。\
「確認いたします」「担当より回答いたします」は問題ありませんが、\
「必ず対応可能です」「確実に実装できます」のように不確実な事柄を断言している場合はNGです。

3. 競合他社への言及: 他社の製品名、サービス名、会社名に具体的に言及している場合。\
自社サービスとの比較や、他社への誘導も含みます。

上記のいずれにも該当しない場合は「pass」と判定してください。
</forbidden_items>"""

_USER_PROMPT_TEMPLATE = """\
以下のカスタマーサポート回答メールを評価してください。

<inquiry>
<content>
{content}
</content>
</inquiry>

<response_email>
{response_body}
</response_email>"""


class ForbiddenContentJudgeResult(BaseModel):
    """禁止コンテンツ評価結果"""

    judgment: Literal["OK", "NG"] = Field(
        description="禁止コンテンツが含まれていなければ OK、含まれていれば NG"
    )
    reason: str = Field(description="判定理由の簡潔な説明")


def forbidden_content_judge(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    """回答メールに禁止コンテンツが含まれていないかを LLM で評価する。"""
    model = get_model(thinking=True)
    model_with_structure = model.with_structured_output(ForbiddenContentJudgeResult)

    if "response_body" not in outputs:
        return {"key": "forbidden_content_judge", "score": None, "comment": "skip: no response_body"}

    user_content = _USER_PROMPT_TEMPLATE.format(
        content=inputs["content"],
        response_body=outputs["response_body"],
    )
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT_TEMPLATE),
        HumanMessage(content=user_content),
    ]
    result: ForbiddenContentJudgeResult = model_with_structure.invoke(messages)  # type: ignore[assignment]

    score = 1 if result.judgment == "OK" else 0
    return {"key": "forbidden_content_judge", "score": score, "comment": result.reason}
