import operator
from typing import Annotated, Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from single_path_plan_generation.main import DecomposedTasks, QueryDecomposer


class Role(BaseModel):
    name: str = Field(..., description="役割の名前")
    description: str = Field(..., description="役割の詳細な説明")
    key_skills: list[str] = Field(..., description="この役割に必要な主要なスキルや属性")


class Task(BaseModel):
    description: str = Field(..., description="タスクの説明")
    role: Role = Field(default=None, description="タスクに割り当てられた役割")


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
        return [Task(description=task) for task in decomposed_tasks.values]


class RoleAssigner:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(TasksWithRoles)

    def run(self, tasks: list[Task]) -> list[Task]:
        prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    (
                        "あなたは創造的な役割設計の専門家です。与えられたタスクに対して、ユニークで適切な役割を生成してください。"
                    ),
                ),
                (
                    "human",
                    (
                        "タスク:\n{tasks}\n\n"
                        "これらのタスクに対して、以下の指示に従って役割を割り当ててください：\n"
                        "1. 各タスクに対して、独自の創造的な役割を考案してください。既存の職業名や一般的な役割名にとらわれる必要はありません。\n"
                        "2. 役割名は、そのタスクの本質を反映した魅力的で記憶に残るものにしてください。\n"
                        "3. 各役割に対して、その役割がなぜそのタスクに最適なのかを説明する詳細な説明を提供してください。\n"
                        "4. その役割が効果的にタスクを遂行するために必要な主要なスキルやアトリビュートを3つ挙げてください。\n\n"
                        "創造性を発揮し、タスクの本質を捉えた革新的な役割を生成してください。"
                    ),
                ),
            ],
        )
        chain = prompt | self.llm
        tasks_with_roles = chain.invoke(
            {"tasks": "\n".join([task.description for task in tasks])}
        )
        return tasks_with_roles.tasks


class Executor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.tools = [TavilySearchResults(max_results=3)]
        self.base_agent = create_react_agent(self.llm, self.tools)

    def run(self, task: Task, results: list[str]) -> str:
        results_str = "\n\n".join(
            f"Info {i+1}:\n{result}" for i, result in enumerate(results)
        )
        result = self.base_agent.invoke(
            {
                "messages": [
                    (
                        "system",
                        (
                            f"あなたは{task.role.name}です。\n"
                            f"説明: {task.role.description}\n"
                            f"主要なスキル: {', '.join(task.role.key_skills)}\n"
                            "あなたの役割に基づいて、与えられたタスクを最高の能力で遂行してください。"
                        ),
                    ),
                    (
                        "human",
                        f"以下のタスクを実行してください：\n\n{task.description}\n\n"
                        f"ここまでのタスクの実行結果:\n{results_str}",
                    ),
                ]
            }
        )
        return result["messages"][-1].content


class Reporter:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, query: str, results: list[str]) -> str:
        prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    (
                        "あなたは総合的なレポート作成の専門家です。複数の情報源からの結果を統合し、洞察力に富んだ包括的なレポートを作成する能力があります。"
                    ),
                ),
                (
                    "human",
                    (
                        "タスク: 以下の情報に基づいて、包括的で一貫性のある回答を作成してください。\n"
                        "要件:\n"
                        "1. 提供されたすべての情報を統合し、よく構成された回答にしてください。\n"
                        "2. 回答は元のクエリに直接応える形にしてください。\n"
                        "3. 各情報の重要なポイントや発見を含めてください。\n"
                        "4. 最後に結論や要約を提供してください。\n"
                        "5. 回答は詳細でありながら簡潔にし、250〜300語程度を目指してください。\n"
                        "6. 回答は日本語で行ってください。\n\n"
                        "ユーザーの依頼: {query}\n\n"
                        "収集した情報:\n{results}"
                    ),
                ),
            ],
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "query": query,
                "results": "\n\n".join(
                    f"Info {i+1}:\n{result}" for i, result in enumerate(results)
                ),
            }
        )


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

    from settings import Settings

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
