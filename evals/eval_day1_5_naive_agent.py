import asyncio
from typing import Any

import weave
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from weave import Evaluation, Model

from pages.day1_5_naive_agent import create_agent_with_tools


class MyModel(Model):
    model_name: str
    reasoning_effort: str

    @weave.op
    def predict(self, message: str) -> Any:
        # エージェントを作成
        agent = create_agent_with_tools(
            model_name=self.model_name,
            reasoning_effort=self.reasoning_effort,
        )

        # エージェントを実行
        return agent.invoke({"messages": [HumanMessage(content=message)]})


@weave.op
def exist_tool_call(output: Any, expected_tool: str) -> int:
    messages = output["messages"]

    # expected_toolが含まれるtool_callがあるかどうかを確認
    for message in messages:
        if isinstance(message, AIMessage):
            if len(message.tool_calls) > 0:
                for tool_call in message.tool_calls:
                    if tool_call["name"] == expected_tool:
                        return 1

    return 0


dataset = [
    {"message": "東京の今日の天気は？", "expected_tool": "tavily_search"},
    {"message": "東京の今日の天気は？Web検索して", "expected_tool": "tavily_search"},
]


def main() -> None:
    load_dotenv(override=True)
    weave.init("training-llm-app")

    # 評価対象の処理を準備
    my_model = MyModel(
        model_name="gpt-5-nano",
        reasoning_effort="minimal",
    )

    # 評価の実行
    evaluation = Evaluation(
        dataset=dataset,
        scorers=[exist_tool_call],
    )
    asyncio.run(evaluation.evaluate(my_model))


if __name__ == "__main__":
    main()
