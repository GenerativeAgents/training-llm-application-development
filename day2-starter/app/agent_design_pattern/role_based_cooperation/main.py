import operator
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.agent_design_pattern.single_path_plan_generation.main import (
    DecomposedTasks,
    QueryDecomposer,
)


class Role(BaseModel):
    name: str = Field(..., description="役割の名前")
    description: str = Field(..., description="役割の詳細な説明")
    key_skills: list[str] = Field(..., description="この役割に必要な主要なスキルや属性")


class Task(BaseModel):
    description: str = Field(..., description="タスクの説明")
    role: Role | None = Field(..., description="タスクに割り当てられた役割")


class TasksWithRoles(BaseModel):
    tasks: list[Task] = Field(..., description="役割が割り当てられたタスクのリスト")


class AgentState(BaseModel):
    query: str = Field(..., description="ユーザーが入力したクエリ")
    tasks: list[Task] = Field(
        default_factory=list, description="実行するタスクのリスト"
    )
    current_task_index: int = Field(default=0, description="現在実行中のタスクの番号")
    results: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="実行済みタスクの結果リスト"
    )
    final_report: str = Field(default="", description="最終的な出力結果")


class Planner:
    def __init__(self, llm: ChatOpenAI):
        self.query_decomposer = QueryDecomposer(llm=llm)

    def run(self, query: str) -> list[Task]:
        decomposed_tasks: DecomposedTasks = self.query_decomposer.run(query=query)
        return [Task(description=task, role=None) for task in decomposed_tasks.tasks]


_role_assigner_system_prompt = "あなたは創造的な役割設計の専門家です。与えられたタスクに対して、ユニークで適切な役割を生成してください。"

_role_assigner_human_prompt_template = """
タスク:
{tasks}

これらのタスクに対して、以下の指示に従って役割を割り当ててください：
1. 各タスクに対して、独自の創造的な役割を考案してください。既存の職業名や一般的な役割名にとらわれる必要はありません。
2. 役割名は、そのタスクの本質を反映した魅力的で記憶に残るものにしてください。
3. 各役割に対して、その役割がなぜそのタスクに最適なのかを説明する詳細な説明を提供してください。
4. その役割が効果的にタスクを遂行するために必要な主要なスキルやアトリビュートを3つ挙げてください。

創造性を発揮し、タスクの本質を捉えた革新的な役割を生成してください。
""".strip()


class RoleAssigner:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, tasks: list[Task]) -> list[Task]:
        tasks_str = "\n".join([task.description for task in tasks])
        prompt = [
            SystemMessage(content=_role_assigner_system_prompt),
            HumanMessage(
                content=_role_assigner_human_prompt_template.format(tasks=tasks_str)
            ),
        ]
        llm_with_structure = self.llm.with_structured_output(TasksWithRoles)
        tasks_with_roles: TasksWithRoles = llm_with_structure.invoke(prompt)  # type: ignore[assignment]
        return tasks_with_roles.tasks


_executor_system_prompt_template = """
あなたは{role_name}です。
説明: {role_description}
主要なスキル: {role_key_skills}
あなたの役割に基づいて、与えられたタスクを最高の能力で遂行してください。
""".strip()

_executor_human_prompt_template = """
以下のタスクを実行してください：
{task_description}

ここまでのタスクの実行結果:
{results}
""".strip()


class Executor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.tools = [TavilySearch(max_results=3)]

    def run(self, task: Task, results: list[str]) -> str:
        if task.role is None:
            raise ValueError("タスクに役割が割り当てられていません")
        system_prompt = _executor_system_prompt_template.format(
            role_name=task.role.name,
            role_description=task.role.description,
            role_key_skills=", ".join(task.role.key_skills),
        )
        results_str = "\n\n".join(
            f"Info {i + 1}:\n{result}" for i, result in enumerate(results)
        )
        human_prompt = _executor_human_prompt_template.format(
            task_description=task.description, results=results_str
        )

        self.base_agent: CompiledStateGraph = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
        )
        result = self.base_agent.invoke(
            {"messages": [HumanMessage(content=human_prompt)]}
        )
        return result["messages"][-1].content


_reporter_system_prompt = "あなたは総合的なレポート作成の専門家です。複数の情報源からの結果を統合し、洞察力に富んだ包括的なレポートを作成する能力があります。"
_reporter_human_prompt_template = """
タスク: 以下の情報に基づいて、包括的で一貫性のある回答を作成してください。
要件:
1. 提供されたすべての情報を統合し、よく構成された回答にしてください。
2. 回答は元のクエリに直接応える形にしてください。
3. 各情報の重要なポイントや発見を含めてください。
4. 最後に結論や要約を提供してください。
5. 回答は詳細でありながら簡潔にし、250〜300語程度を目指してください。
6. 回答は日本語で行ってください。

ユーザーの依頼: {query}
収集した情報: {results}
""".strip()


class Reporter:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, query: str, results: list[str]) -> str:
        results_str = "\n\n".join(
            f"Info {i + 1}:\n{result}" for i, result in enumerate(results)
        )
        prompt = [
            SystemMessage(content=_reporter_system_prompt),
            HumanMessage(
                content=_reporter_human_prompt_template.format(
                    query=query,
                    results=results_str,
                )
            ),
        ]
        ai_message = self.llm.invoke(prompt)
        return ai_message.content  # type: ignore[return-value]


class RoleBasedCooperation:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.planner = Planner(llm=llm)
        self.role_assigner = RoleAssigner(llm=llm)
        self.executor = Executor(llm=llm)
        self.reporter = Reporter(llm=llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("planner", self._plan_tasks)
        workflow.add_node("role_assigner", self._assign_roles)
        workflow.add_node("executor", self._execute_task)
        workflow.add_node("reporter", self._generate_report)

        workflow.set_entry_point("planner")

        workflow.add_edge("planner", "role_assigner")
        workflow.add_edge("role_assigner", "executor")
        workflow.add_conditional_edges(
            "executor",
            lambda state: state.current_task_index < len(state.tasks),
            {True: "executor", False: "reporter"},
        )

        workflow.add_edge("reporter", END)

        return workflow.compile()

    def _plan_tasks(self, state: AgentState) -> dict[str, Any]:
        tasks = self.planner.run(query=state.query)
        return {"tasks": tasks}

    def _assign_roles(self, state: AgentState) -> dict[str, Any]:
        tasks_with_roles = self.role_assigner.run(tasks=state.tasks)
        return {"tasks": tasks_with_roles}

    def _execute_task(self, state: AgentState) -> dict[str, Any]:
        current_task = state.tasks[state.current_task_index]
        result = self.executor.run(task=current_task, results=state.results)
        return {
            "results": [result],
            "current_task_index": state.current_task_index + 1,
        }

    def _generate_report(self, state: AgentState) -> dict[str, Any]:
        report = self.reporter.run(query=state.query, results=state.results)
        return {"final_report": report}

    def run(self, query: str) -> str:
        initial_state = AgentState(query=query)
        final_state = self.graph.invoke(initial_state, {"recursion_limit": 1000})
        return final_state["final_report"]


def main():
    import argparse

    from app.agent_design_pattern.settings import Settings

    settings = Settings()
    parser = argparse.ArgumentParser(
        description="RoleBasedCooperationを使用してタスクを実行します"
    )
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()

    llm = ChatOpenAI(
        model=settings.openai_smart_model, temperature=settings.temperature
    )
    agent = RoleBasedCooperation(llm=llm)
    result = agent.run(query=args.task)
    print(result)


if __name__ == "__main__":
    main()
