from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agent_design_pattern.passive_goal_creator.main import Goal, PassiveGoalCreator


class OptimizedGoal(BaseModel):
    description: str = Field(..., description="目標の説明")
    metrics: str = Field(..., description="目標の達成度を測定する方法")

    @property
    def text(self) -> str:
        return f"{self.description}(測定基準: {self.metrics})"


_prompt_template = """
あなたは目標設定の専門家です。以下の目標をSMART原則（Specific: 具体的、Measurable: 測定可能、Achievable: 達成可能、Relevant: 関連性が高い、Time-bound: 期限がある）に基づいて最適化してください。

元の目標:
{query}

指示:
1. 元の目標を分析し、不足している要素や改善点を特定してください。
2. あなたが実行可能な行動は以下の行動だけです。
   - インターネットを利用して、目標を達成するための調査を行う。
   - ユーザーのためのレポートを生成する。
3. SMART原則の各要素を考慮しながら、目標を具体的かつ詳細に記載してください。
   - 一切抽象的な表現を含んではいけません。
   - 必ず全ての単語が実行可能かつ具体的であることを確認してください。
4. 目標の達成度を測定する方法を具体的かつ詳細に記載してください。
5. 元の目標で期限が指定されていない場合は、期限を考慮する必要はありません。
6. REMEMBER: 決して2.以外の行動を取ってはいけません。
""".strip()


class PromptOptimizer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, query: str) -> OptimizedGoal:
        prompt = _prompt_template.format(query=query)
        model_with_structure = self.llm.with_structured_output(OptimizedGoal)
        return model_with_structure.invoke(prompt)  # type: ignore[return-value]


def main():
    import argparse

    from app.agent_design_pattern.settings import Settings

    settings = Settings()

    parser = argparse.ArgumentParser(
        description="PromptOptimizerを利用して、生成された目標のリストを最適化します"
    )
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()

    llm = ChatOpenAI(
        model=settings.openai_smart_model, temperature=settings.temperature
    )

    passive_goal_creator = PassiveGoalCreator(llm=llm)
    goal: Goal = passive_goal_creator.run(query=args.task)

    prompt_optimizer = PromptOptimizer(llm=llm)
    optimised_goal: OptimizedGoal = prompt_optimizer.run(query=goal.text)

    print(f"{optimised_goal.text}")


if __name__ == "__main__":
    main()
