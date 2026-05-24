from typing import Literal, TypedDict

TopicType = Literal["development", "product", "other", "spam"]
QualityJudgment = Literal["OK", "NG"]


class InputState(TypedDict):
    # 入力
    customer_name: str
    company_name: str | None
    content: str


class GraphState(TypedDict, total=False):
    # 入力
    customer_name: str
    company_name: str | None
    content: str
    # classify_topic 出力
    topic: TopicType
    classification_confidence: float
    # generate_response 出力
    response_subject: str
    response_body: str
    # quality_check 出力
    politeness: QualityJudgment
    politeness_reason: str
