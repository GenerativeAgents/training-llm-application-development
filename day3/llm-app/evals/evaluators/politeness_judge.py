from typing import Any, Literal

import weave
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm import get_model

_SYSTEM_PROMPT_TEMPLATE = """\
あなたはビジネス日本語の品質を評価する専門家です。
カスタマーサポートの回答メールが、適切なビジネス敬語で書かれているかを評価してください。

<criteria>
以下のすべてを満たす場合に「polite」と判定してください。

1. 適切な敬語（尊敬語・謙譲語・丁寧語）が正しく使い分けられている
2. ビジネスメールとして適切な挨拶文・締めの挨拶が含まれている
3. 顧客に対して失礼な表現、命令口調、タメ口が含まれていない
4. 全体を通して丁寧で誠実な印象を与える文面である

上記のいずれかを満たさない場合は「not_polite」と判定してください。
</criteria>"""

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


class PolitenessJudgeResult(BaseModel):
    """敬語・丁寧さ評価結果"""

    judgment: Literal["OK", "NG"] = Field(
        description="丁寧なビジネス日本語かどうかの判定"
    )
    reason: str = Field(description="判定理由の簡潔な説明")


@weave.op()
def politeness_judge(output: dict[str, Any], content: str) -> dict[str, Any]:
    """回答メールの敬語・丁寧さを LLM で評価する。"""
    if "response_body" not in output:
        return {"key": "politeness_judge", "score": None, "comment": "skip: no response_body"}

    model = get_model(thinking=True)
    model_with_structure = model.with_structured_output(PolitenessJudgeResult)

    user_content = _USER_PROMPT_TEMPLATE.format(
        content=content,
        response_body=output["response_body"],
    )
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT_TEMPLATE),
        HumanMessage(content=user_content),
    ]
    result: PolitenessJudgeResult = model_with_structure.invoke(messages)  # type: ignore[assignment]

    score = 1 if result.judgment == "OK" else 0
    return {"key": "politeness_judge", "score": score, "comment": result.reason}
