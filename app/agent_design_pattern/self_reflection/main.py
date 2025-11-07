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

from app.agent_design_pattern.common.reflection_manager import (
    Reflection,
    ReflectionManager,
    TaskReflector,
)
from app.agent_design_pattern.passive_goal_creator.main import Goal, PassiveGoalCreator
from app.agent_design_pattern.prompt_optimizer.main import (
    OptimizedGoal,
    PromptOptimizer,
)
from app.agent_design_pattern.response_optimizer.main import ResponseOptimizer


def format_reflections(reflections: list[Reflection]) -> str:
    return (
        "\n\n".join(
            f"<ref_{i}><task>{r.task}</task><reflection>{r.reflection}</reflection></ref_{i}>"
            for i, r in enumerate(reflections)
        )
        if reflections
        else "No relevant past reflections."
    )


class DecomposedTasks(BaseModel):
    tasks: list[str] = Field(
        default_factory=list,
        description="3~5個に分解されたタスク",
    )


class ReflectiveAgentState(BaseModel):
    query: str = Field(..., description="ユーザーが最初に入力したクエリ")
    optimized_goal: str = Field(default="", description="最適化された目標")
    optimized_response: str = Field(
        default="", description="最適化されたレスポンス定義"
    )
    tasks: list[str] = Field(default_factory=list, description="実行するタスクのリスト")
    current_task_index: int = Field(default=0, description="現在実行中のタスクの番号")
    results: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="実行済みタスクの結果リスト"
    )
    reflection_ids: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="リフレクション結果のIDリスト"
    )
    final_output: str = Field(default="", description="最終的な出力結果")
    retry_count: int = Field(default=0, description="タスクの再試行回数")


class ReflectiveGoalCreator:
    def __init__(self, llm: ChatOpenAI, reflection_manager: ReflectionManager):
        self.llm = llm
        self.reflection_manager = reflection_manager
        self.passive_goal_creator = PassiveGoalCreator(llm=self.llm)
        self.prompt_optimizer = PromptOptimizer(llm=self.llm)

    def run(self, query: str) -> str:
        relevant_reflections = self.reflection_manager.get_relevant_reflections(query)
        reflection_text = format_reflections(relevant_reflections)

        query = f"{query}\n\n目標設定する際に以下の過去のふりかえりを考慮すること:\n{reflection_text}"
        goal: Goal = self.passive_goal_creator.run(query=query)
        optimized_goal: OptimizedGoal = self.prompt_optimizer.run(query=goal.text)
        return optimized_goal.text


class ReflectiveResponseOptimizer:
    def __init__(self, llm: ChatOpenAI, reflection_manager: ReflectionManager):
        self.llm = llm
        self.reflection_manager = reflection_manager
        self.response_optimizer = ResponseOptimizer(llm=llm)

    def run(self, query: str) -> str:
        relevant_reflections = self.reflection_manager.get_relevant_reflections(query)
        reflection_text = format_reflections(relevant_reflections)

        query = f"{query}\n\nレスポンス最適化に以下の過去のふりかえりを考慮すること:\n{reflection_text}"
        optimized_response: str = self.response_optimizer.run(query=query)
        return optimized_response


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
5. タスクを作成する際に以下の過去のふりかえりを考慮すること:
{reflections}

目標: {query}
""".strip()


class QueryDecomposer:
    def __init__(self, llm: ChatOpenAI, reflection_manager: ReflectionManager):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.reflection_manager = reflection_manager

    def run(self, query: str) -> DecomposedTasks:
        relevant_reflections = self.reflection_manager.get_relevant_reflections(query)
        reflection_text = format_reflections(relevant_reflections)
        prompt = _query_decomposer_prompt_template.format(
            current_date=self.current_date,
            reflections=reflection_text,
            query=query,
        )
        llm_with_structure = self.llm.with_structured_output(DecomposedTasks)
        return llm_with_structure.invoke(prompt)  # type: ignore[return-value]


_task_executor_prompt_template = """
CURRENT_DATE: {current_date}
-----
次のタスクを実行し、詳細な回答を提供してください。

タスク: {task}

要件:
1. 必要に応じて提供されたツールを使用すること。
2. 実行において徹底的かつ包括的であること。
3. 可能な限り具体的な事実やデータを提供すること。
4. 発見事項を明確に要約すること。
5. 以下の過去のふりかえりを考慮すること:
{reflection_text}

ここまでのタスクの実行結果:
{results_str}
"""


class TaskExecutor:
    def __init__(self, llm: ChatOpenAI, reflection_manager: ReflectionManager):
        self.llm = llm
        self.reflection_manager = reflection_manager
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.tools = [TavilySearch(max_results=3)]

    def run(self, task: str, results: list[str]) -> str:
        relevant_reflections = self.reflection_manager.get_relevant_reflections(task)
        reflection_text = format_reflections(relevant_reflections)
        agent: CompiledStateGraph = create_agent(model=self.llm, tools=self.tools)
        results_str = "\n\n".join(
            f"Info {i + 1}:\n{result}" for i, result in enumerate(results)
        )
        prompt = _task_executor_prompt_template.format(
            current_date=self.current_date,
            task=task,
            reflection_text=reflection_text,
            results_str=results_str,
        )
        result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        return result["messages"][-1].content


_result_aggregator_prompt_template = """
与えられた目標:
{query}

調査結果:
{results}

与えられた目標に対し、調査結果を用いて、以下の指示に基づいてレスポンスを生成してください。
{response_definition}

過去のふりかえりを考慮すること:
{reflection_text}
"""


class ResultAggregator:
    def __init__(self, llm: ChatOpenAI, reflection_manager: ReflectionManager):
        self.llm = llm
        self.reflection_manager = reflection_manager
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(
        self,
        query: str,
        results: list[str],
        reflection_ids: list[str],
        response_definition: str,
    ) -> str:
        relevant_reflections = [
            self.reflection_manager.get_reflection(rid) for rid in reflection_ids
        ]
        results_str = "\n\n".join(
            f"Info {i + 1}:\n{result}" for i, result in enumerate(results)
        )
        prompt = _result_aggregator_prompt_template.format(
            current_date=self.current_date,
            query=query,
            results=results_str,
            response_definition=response_definition,
            reflection_text=format_reflections(relevant_reflections),
        )
        ai_message = self.llm.invoke(prompt)
        return ai_message.content  # type: ignore[return-value]


class ReflectiveAgent:
    def __init__(
        self,
        llm: ChatOpenAI,
        reflection_manager: ReflectionManager,
        task_reflector: TaskReflector,
        max_retries: int = 2,
    ):
        self.reflection_manager = reflection_manager
        self.task_reflector = task_reflector
        self.reflective_goal_creator = ReflectiveGoalCreator(
            llm=llm, reflection_manager=self.reflection_manager
        )
        self.reflective_response_optimizer = ReflectiveResponseOptimizer(
            llm=llm, reflection_manager=self.reflection_manager
        )
        self.query_decomposer = QueryDecomposer(
            llm=llm, reflection_manager=self.reflection_manager
        )
        self.task_executor = TaskExecutor(
            llm=llm, reflection_manager=self.reflection_manager
        )
        self.result_aggregator = ResultAggregator(
            llm=llm, reflection_manager=self.reflection_manager
        )
        self.max_retries = max_retries
        self.graph = self._create_graph()

    def _create_graph(self) -> CompiledStateGraph:
        graph = StateGraph(ReflectiveAgentState)
        graph.add_node("goal_setting", self._goal_setting)
        graph.add_node("decompose_query", self._decompose_query)
        graph.add_node("execute_task", self._execute_task)
        graph.add_node("reflect_on_task", self._reflect_on_task)
        graph.add_node("update_task_index", self._update_task_index)
        graph.add_node("aggregate_results", self._aggregate_results)
        graph.set_entry_point("goal_setting")
        graph.add_edge("goal_setting", "decompose_query")
        graph.add_edge("decompose_query", "execute_task")
        graph.add_edge("execute_task", "reflect_on_task")
        graph.add_conditional_edges(
            "reflect_on_task",
            self._should_retry_or_continue,
            {
                "retry": "execute_task",
                "continue": "update_task_index",
                "finish": "aggregate_results",
            },
        )
        graph.add_edge("update_task_index", "execute_task")
        graph.add_edge("aggregate_results", END)
        return graph.compile()

    def _goal_setting(self, state: ReflectiveAgentState) -> dict[str, Any]:
        optimized_goal: str = self.reflective_goal_creator.run(query=state.query)
        optimized_response: str = self.reflective_response_optimizer.run(
            query=optimized_goal
        )
        return {
            "optimized_goal": optimized_goal,
            "optimized_response": optimized_response,
        }

    def _decompose_query(self, state: ReflectiveAgentState) -> dict[str, Any]:
        tasks: DecomposedTasks = self.query_decomposer.run(query=state.optimized_goal)
        return {"tasks": tasks.tasks}

    def _execute_task(self, state: ReflectiveAgentState) -> dict[str, Any]:
        current_task = state.tasks[state.current_task_index]
        result = self.task_executor.run(task=current_task, results=state.results)
        return {"results": [result], "current_task_index": state.current_task_index}

    def _reflect_on_task(self, state: ReflectiveAgentState) -> dict[str, Any]:
        current_task = state.tasks[state.current_task_index]
        current_result = state.results[-1]
        reflection = self.task_reflector.run(task=current_task, result=current_result)
        return {
            "reflection_ids": [reflection.id],
            "retry_count": (
                state.retry_count + 1 if reflection.judgment.needs_retry else 0
            ),
        }

    def _should_retry_or_continue(self, state: ReflectiveAgentState) -> str:
        latest_reflection_id = state.reflection_ids[-1]
        latest_reflection = self.reflection_manager.get_reflection(latest_reflection_id)
        if (
            latest_reflection
            and latest_reflection.judgment.needs_retry
            and state.retry_count < self.max_retries
        ):
            return "retry"
        elif state.current_task_index < len(state.tasks) - 1:
            return "continue"
        else:
            return "finish"

    def _update_task_index(self, state: ReflectiveAgentState) -> dict[str, Any]:
        return {"current_task_index": state.current_task_index + 1}

    def _aggregate_results(self, state: ReflectiveAgentState) -> dict[str, Any]:
        final_output = self.result_aggregator.run(
            query=state.optimized_goal,
            results=state.results,
            reflection_ids=state.reflection_ids,
            response_definition=state.optimized_response,
        )
        return {"final_output": final_output}

    def run(self, query: str) -> str:
        initial_state = ReflectiveAgentState(query=query)
        final_state = self.graph.invoke(initial_state, {"recursion_limit": 1000})
        return final_state.get("final_output", "エラー: 出力に失敗しました。")


def main():
    import argparse

    from app.agent_design_pattern.settings import Settings

    settings = Settings()

    parser = argparse.ArgumentParser(
        description="ReflectiveAgentを使用してタスクを実行します（Self-reflection）"
    )
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()

    llm = ChatOpenAI(
        model=settings.openai_smart_model, temperature=settings.temperature
    )
    reflection_manager = ReflectionManager(file_path="tmp/self_reflection_db.json")
    task_reflector = TaskReflector(llm=llm, reflection_manager=reflection_manager)
    agent = ReflectiveAgent(
        llm=llm, reflection_manager=reflection_manager, task_reflector=task_reflector
    )
    result = agent.run(args.task)
    print(result)


if __name__ == "__main__":
    main()
