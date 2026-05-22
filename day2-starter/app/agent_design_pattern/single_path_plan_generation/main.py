import operator
from datetime import datetime
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.agent_design_pattern.passive_goal_creator.main import Goal, PassiveGoalCreator
from app.agent_design_pattern.prompt_optimizer.main import (
    OptimizedGoal,
    PromptOptimizer,
)
from app.agent_design_pattern.response_optimizer.main import ResponseOptimizer


class DecomposedTasks(BaseModel):
    tasks: list[str] = Field(
        default_factory=list,
        description="3~5個に分解されたタスク",
    )


class SinglePathPlanGenerationState(BaseModel):
    query: str = Field(..., description="ユーザーが入力したクエリ")
    optimized_goal: str = Field(default="", description="最適化された目標")
    optimized_response: str = Field(
        default="", description="最適化されたレスポンス定義"
    )
    tasks: list[str] = Field(default_factory=list, description="実行するタスクのリスト")
    current_task_index: int = Field(default=0, description="現在実行中のタスクの番号")
    results: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="実行済みタスクの結果リスト"
    )
    final_output: str = Field(default="", description="最終的な出力結果")


_query_decomposer_prompt_template = """
CURRENT_DATE: {current_date}
-----
タスク: 与えられた目標を具体的で実行可能なタスクに分解してください。
要件:
1. 以下の行動だけで目標を達成すること。決して指定された以外の行動をとらないこと。
   - インターネットを利用して、目標を達成するための調査を行う。
2. 各タスクは具体的かつ詳細に記載されており、単独で実行ならびに検証可能な情報を含めること。一切抽象的な表現を含まないこと。
3. タスクは実行可能な順序でリスト化すること。
4. タスクは日本語で出力すること。
目標: {query}
""".strip()


class QueryDecomposer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(self, query: str) -> DecomposedTasks:
        prompt = _query_decomposer_prompt_template.format(
            current_date=self.current_date,
            query=query,
        )
        model_with_structure = self.llm.with_structured_output(DecomposedTasks)
        return model_with_structure.invoke(prompt)  # type: ignore[return-value]


_task_executor_prompt_template = """
次のタスクを実行し、詳細な回答を提供してください。

タスク: {task}

要件:
1. 必要に応じて提供されたツールを使用してください。
2. 実行は徹底的かつ包括的に行ってください。
3. 可能な限り具体的な事実やデータを提供してください。
4. 発見した内容を明確に要約してください。

ここまでのタスクの実行結果:
{results_str}
""".strip()


class TaskExecutor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.tools = [TavilySearch(max_results=3)]

    def run(self, task: str, results: list[str]) -> str:
        agent: CompiledStateGraph = create_agent(model=self.llm, tools=self.tools)
        results_str = "\n\n".join(
            f"Info {i + 1}:\n{result}" for i, result in enumerate(results)
        )
        prompt = _task_executor_prompt_template.format(
            task=task,
            results_str=results_str,
        )
        result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        return result["messages"][-1].content


_result_aggregator_prompt_template = """
与えられた目標: {query}

調査結果: {results}

与えられた目標に対し、調査結果を用いて、以下の指示に基づいてレスポンスを生成してください。
{response_definition}
""".strip()


class ResultAggregator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, query: str, response_definition: str, results: list[str]) -> str:
        results_str = "\n\n".join(
            f"Info {i + 1}:\n{result}" for i, result in enumerate(results)
        )
        prompt = _result_aggregator_prompt_template.format(
            query=query,
            results=results_str,
            response_definition=response_definition,
        )
        ai_message = self.llm.invoke(prompt)
        return ai_message.content  # type: ignore[return-value]


class SinglePathPlanGeneration:
    def __init__(self, llm: ChatOpenAI):
        self.passive_goal_creator = PassiveGoalCreator(llm=llm)
        self.prompt_optimizer = PromptOptimizer(llm=llm)
        self.response_optimizer = ResponseOptimizer(llm=llm)
        self.query_decomposer = QueryDecomposer(llm=llm)
        self.task_executor = TaskExecutor(llm=llm)
        self.result_aggregator = ResultAggregator(llm=llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> CompiledStateGraph:
        graph = StateGraph(SinglePathPlanGenerationState)
        graph.add_node("goal_setting", self._goal_setting)
        graph.add_node("decompose_query", self._decompose_query)
        graph.add_node("execute_task", self._execute_task)
        graph.add_node("aggregate_results", self._aggregate_results)
        graph.set_entry_point("goal_setting")
        graph.add_edge("goal_setting", "decompose_query")
        graph.add_edge("decompose_query", "execute_task")
        graph.add_conditional_edges(
            "execute_task",
            lambda state: state.current_task_index < len(state.tasks),
            {True: "execute_task", False: "aggregate_results"},
        )
        graph.add_edge("aggregate_results", END)
        return graph.compile()

    def _goal_setting(self, state: SinglePathPlanGenerationState) -> dict[str, Any]:
        # プロンプト最適化
        goal: Goal = self.passive_goal_creator.run(query=state.query)
        optimized_goal: OptimizedGoal = self.prompt_optimizer.run(query=goal.text)
        # レスポンス最適化
        optimized_response: str = self.response_optimizer.run(query=optimized_goal.text)
        return {
            "optimized_goal": optimized_goal.text,
            "optimized_response": optimized_response,
        }

    def _decompose_query(self, state: SinglePathPlanGenerationState) -> dict[str, Any]:
        decomposed_tasks: DecomposedTasks = self.query_decomposer.run(
            query=state.optimized_goal
        )
        return {"tasks": decomposed_tasks.tasks}

    def _execute_task(self, state: SinglePathPlanGenerationState) -> dict[str, Any]:
        current_task = state.tasks[state.current_task_index]
        result = self.task_executor.run(task=current_task, results=state.results)
        return {
            "results": [result],
            "current_task_index": state.current_task_index + 1,
        }

    def _aggregate_results(
        self, state: SinglePathPlanGenerationState
    ) -> dict[str, Any]:
        final_output = self.result_aggregator.run(
            query=state.optimized_goal,
            response_definition=state.optimized_response,
            results=state.results,
        )
        return {"final_output": final_output}

    def run(self, query: str) -> str:
        initial_state = SinglePathPlanGenerationState(query=query)
        final_state = self.graph.invoke(initial_state, {"recursion_limit": 1000})
        return final_state.get("final_output", "Failed to generate a final response.")


def main():
    import argparse

    from app.agent_design_pattern.settings import Settings

    settings = Settings()

    parser = argparse.ArgumentParser(
        description="SinglePathPlanGenerationを使用してタスクを実行します"
    )
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()

    llm = ChatOpenAI(
        model=settings.openai_smart_model, temperature=settings.temperature
    )
    agent = SinglePathPlanGeneration(llm=llm)
    result = agent.run(args.task)
    print(result)


if __name__ == "__main__":
    main()
