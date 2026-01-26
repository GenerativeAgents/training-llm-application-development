"""
Documentation Agent 評価用スコアラー

3種類のスコアラーを提供:
- SectionCompletenessScorer: 7セクションが全て含まれているか（ルールベース）
- RequirementsCoverageScorer: キー要件がカバーされているか（キーワードマッチ）
- DocumentQualityScorer: 品質評価（LLM-as-judge）
"""

from typing import Any

import weave
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.prompts.documentation_quality_prompt import document_quality_evaluation_prompt

# 要件定義書の7セクション
REQUIRED_SECTIONS = [
    "プロジェクト概要",
    "主要機能",
    "非機能要件",
    "制約条件",
    "ターゲットユーザー",
    "優先順位",
    "リスクと軽減策",
]


class SectionCompletenessScorer(weave.Scorer):
    """
    要件定義書の7セクションが全て含まれているかを評価

    ルールベースで各セクションの存在を確認し、
    存在するセクション数 / 7 をスコアとして返す
    """

    @weave.op
    async def score(self, output: dict[str, Any]) -> dict:
        requirements_doc = output.get("requirements_doc", "")

        present_sections = []
        missing_sections = []

        for section in REQUIRED_SECTIONS:
            if section in requirements_doc:
                present_sections.append(section)
            else:
                missing_sections.append(section)

        completeness = len(present_sections) / len(REQUIRED_SECTIONS)

        return {
            "completeness": completeness,
            "present_sections": present_sections,
            "missing_sections": missing_sections,
            "score": completeness,
        }


class RequirementsCoverageScorer(weave.Scorer):
    """
    キー要件がカバーされているかを評価

    データセットのkey_requirements（カンマ区切り）と
    生成された要件定義書を比較し、キーワードマッチ率を返す
    """

    @weave.op
    async def score(self, output: dict[str, Any], key_requirements: str) -> dict:
        requirements_doc = output.get("requirements_doc", "")

        # カンマ区切りのキーワードをリストに変換
        keywords = [kw.strip() for kw in key_requirements.split(",") if kw.strip()]

        covered = []
        not_covered = []

        for keyword in keywords:
            if keyword in requirements_doc:
                covered.append(keyword)
            else:
                not_covered.append(keyword)

        coverage = len(covered) / len(keywords) if keywords else 0.0

        return {
            "coverage": coverage,
            "covered_keywords": covered,
            "not_covered_keywords": not_covered,
            "score": coverage,
        }


class QualityEvaluation(BaseModel):
    """ドキュメント品質評価の結果"""

    comprehensiveness_score: float = Field(
        ..., ge=0.0, le=1.0, description="網羅性スコア（0.0-1.0）"
    )
    comprehensiveness_reason: str = Field(..., description="網羅性の評価理由")

    specificity_score: float = Field(
        ..., ge=0.0, le=1.0, description="具体性スコア（0.0-1.0）"
    )
    specificity_reason: str = Field(..., description="具体性の評価理由")

    consistency_score: float = Field(
        ..., ge=0.0, le=1.0, description="整合性スコア（0.0-1.0）"
    )
    consistency_reason: str = Field(..., description="整合性の評価理由")


class DocumentQualityScorer(weave.Scorer):
    """
    ドキュメント品質をLLM-as-judgeで評価

    3つの観点で0.0-1.0スコアを付与:
    - 網羅性: ユーザー要求に必要な情報が網羅されているか
    - 具体性: 実装可能なレベルで具体的か
    - 整合性: セクション間で矛盾がないか
    """

    model_id: str = "gpt-4.1-nano"

    @weave.op
    async def score(
        self, output: dict[str, Any], user_request: str, expected_output: str
    ) -> dict:
        requirements_doc = output.get("requirements_doc", "")

        prompt = ChatPromptTemplate.from_template(
            document_quality_evaluation_prompt.content
        )

        model = init_chat_model(model=self.model_id, model_provider="openai")
        llm_with_structure = model.with_structured_output(QualityEvaluation)

        prompt_value = prompt.invoke(
            {
                "user_request": user_request,
                "expected_output": expected_output,
                "requirements_doc": requirements_doc,
            }
        )

        result: QualityEvaluation = llm_with_structure.invoke(prompt_value)

        # 3つのスコアの平均を総合スコアとする
        average_score = (
            result.comprehensiveness_score
            + result.specificity_score
            + result.consistency_score
        ) / 3

        return {
            "comprehensiveness": {
                "score": result.comprehensiveness_score,
                "reason": result.comprehensiveness_reason,
            },
            "specificity": {
                "score": result.specificity_score,
                "reason": result.specificity_reason,
            },
            "consistency": {
                "score": result.consistency_score,
                "reason": result.consistency_reason,
            },
            "score": average_score,
        }
