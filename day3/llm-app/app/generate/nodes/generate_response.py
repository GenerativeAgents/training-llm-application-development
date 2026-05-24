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
</rules>

<厳守事項>
以下は会社のコンプライアンス上、回答メールに絶対に含めてはいけません。違反すると回答は却下されます。

1. 金額・費用の数値の提示
   「月額10万円」「開発費用は300万〜500万円程度」のように、具体的な金額・料金・費用の数値を一切書かないでください（概算・目安・レンジであっても禁止）。
   費用に関するご質問には金額には触れず、「ご要件を伺ったうえで個別にお見積もりをご提示いたします」のように、お見積もり手続きのご案内に留めてください。

2. 未確定・未確認事項の断定
   確認や個別判断が必要な事柄を断定しないでください。
   「必ず対応可能です」「確実に実装できます」「十分に構築可能です」のような断言や、確認していない実績・仕様・スケジュール・対応可否の断定は避け、
   「詳しくお伺いしたうえで対応可否を検討させていただきます」「担当者より確認のうえご回答いたします」のように表現してください。

3. 推測による詳細情報の創作
   設定手順・連携方法・データ同期頻度・認証取得状況・提供価格・連絡先（電話番号・メールアドレス・URL）など、事実を確認できない具体的な情報を推測で書かないでください。
   一般的なご案内に留め、詳細は「別途資料をお送りいたします」「担当者よりご案内いたします」としてください。

4. 他社（競合）の製品名・サービス名・会社名への言及
   特定の他社の製品名・サービス名・会社名を回答本文に書かないでください（自社製品「AgentBoard」および自社名「株式会社サンプルエージェント」は除く）。
   お客様が言及された連携先・ご利用中のツール等であっても、「ご利用中のチャットツール」「ご指定のCRM」「外部サービス」のように一般化して言及し、他社サービスとの比較・推奨・誘導も行わないでください。
</厳守事項>"""

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
