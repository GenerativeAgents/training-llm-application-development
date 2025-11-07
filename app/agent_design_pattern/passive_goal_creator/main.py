from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Goal(BaseModel):
    description: str = Field(..., description="目標の説明")

    @property
    def text(self) -> str:
        return f"{self.description}"


_prompt_template = """
ユーザーの入力を分析し、明確で実行可能な目標を生成してください。
要件:
1. 目標は具体的かつ明確であり、実行可能なレベルで詳細化されている必要があります。
2. あなたが実行可能な行動は以下の行動だけです。
   - インターネットを利用して、目標を達成するための調査を行う。
   - ユーザーのためのレポートを生成する。
3. 決して2.以外の行動を取ってはいけません。
ユーザーの入力: {query}
""".strip()


class PassiveGoalCreator:
    def __init__(
        self,
        llm: ChatOpenAI,
    ):
        self.llm = llm

    def run(self, query: str) -> Goal:
        prompt = _prompt_template.format(query=query)
        model_with_structure = self.llm.with_structured_output(Goal)
        return model_with_structure.invoke(prompt)  # type: ignore[return-value]


def main():
    import argparse

    from app.agent_design_pattern.settings import Settings

    settings = Settings()

    parser = argparse.ArgumentParser(
        description="PassiveGoalCreatorを利用して目標を生成します"
    )
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()

    llm = ChatOpenAI(
        model=settings.openai_smart_model, temperature=settings.temperature
    )
    goal_creator = PassiveGoalCreator(llm=llm)
    result: Goal = goal_creator.run(query=args.task)

    print(f"{result.text}")


if __name__ == "__main__":
    main()
