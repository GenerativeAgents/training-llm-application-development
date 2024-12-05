import json
import os
import uuid
from typing import Optional

import faiss
import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from retry import retry
from settings import Settings

settings = Settings()


class ReflectionJudgment(BaseModel):
    needs_retry: bool = Field(
        description="タスクの実行結果は適切だったと思いますか?あなたの判断を真偽値で示してください。"
    )
    confidence: float = Field(
        description="あなたの判断に対するあなたの自信の度合いを0から1までの小数で示してください。"
    )
    reasons: list[str] = Field(
        description="タスクの実行結果の適切性とそれに対する自信度について、判断に至った理由を簡潔に列挙してください。"
    )


class Reflection(BaseModel):
    id: str = Field(description="リフレクション内容に一意性を与えるためのID")
    task: str = Field(description="ユーザーから与えられたタスクの内容")
    reflection: str = Field(
        description="このタスクに取り組んだ際のあなたの思考プロセスを振り返ってください。何か改善できる点はありましたか? 次に同様のタスクに取り組む際に、より良い結果を出すための教訓を2〜3文程度で簡潔に述べてください。"
    )
    judgment: ReflectionJudgment = Field(description="リトライが必要かどうかの判定")


class ReflectionManager:
    def __init__(self, file_path: str = settings.default_reflection_db_path):
        self.file_path = file_path
        self.embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model)
        self.reflections: dict[str, Reflection] = {}
        self.embeddings_dict: dict[str, list[float]] = {}
        self.index = None
        self.load_reflections()

    def load_reflections(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                data = json.load(file)
                for item in data:
                    reflection = Reflection(**item["reflection"])
                    self.reflections[reflection.id] = reflection
                    self.embeddings_dict[reflection.id] = item["embedding"]

            if self.reflections:
                embeddings = list(self.embeddings_dict.values())
                self.index = faiss.IndexFlatL2(len(embeddings[0]))
                self.index.add(np.array(embeddings).astype("float32"))

    def save_reflection(self, reflection: Reflection) -> str:
        reflection.id = str(uuid.uuid4())
        reflection_id = reflection.id
        self.reflections[reflection_id] = reflection
        embedding = self.embeddings.embed_query(reflection.reflection)
        self.embeddings_dict[reflection_id] = embedding

        if self.index is None:
            self.index = faiss.IndexFlatL2(len(embedding))
        self.index.add(np.array([embedding]).astype("float32"))

        with open(self.file_path, "w", encoding="utf-8") as file:
            json.dump(
                [
                    {"reflection": reflection.dict(), "embedding": embedding}
                    for reflection, embedding in zip(
                        self.reflections.values(), self.embeddings_dict.values()
                    )
                ],
                file,
                ensure_ascii=False,
                indent=4,
            )

        return reflection_id

    def get_reflection(self, reflection_id: str) -> Optional[Reflection]:
        return self.reflections.get(reflection_id)

    def get_relevant_reflections(self, query: str, k: int = 3) -> list[Reflection]:
        if not self.reflections or self.index is None:
            return []

        query_embedding = self.embeddings.embed_query(query)
        try:
            D, I = self.index.search(
                np.array([query_embedding]).astype("float32"),
                min(k, len(self.reflections)),
            )
            reflection_ids = list(self.reflections.keys())
            return [
                self.reflections[reflection_ids[i]]
                for i in I[0]
                if i < len(reflection_ids)
            ]
        except Exception as e:
            print(f"Error during reflection search: {e}")
            return []


class TaskReflector:
    def __init__(self, llm: BaseChatModel, reflection_manager: ReflectionManager):
        self.llm = llm.with_structured_output(Reflection)
        self.reflection_manager = reflection_manager

    def run(self, task: str, result: str) -> Reflection:
        prompt = ChatPromptTemplate.from_template(
            "与えられたタスクの内容:\n{task}\n\n"
            "タスクを実行した結果:\n{result}\n\n"
            "あなたは高度な推論能力を持つAIエージェントです。上記のタスクを実行した結果を分析し、このタスクに対するあなたの取り組みが適切だったかどうかを内省してください。\n"
            "以下の項目に沿って、リフレクションの内容を出力してください。\n\n"
            "リフレクション:\n"
            "このタスクに取り組んだ際のあなたの思考プロセスや方法を振り返ってください。何か改善できる点はありましたか?\n"
            "次に同様のタスクに取り組む際に、より良い結果を出すための教訓を2〜3文程度で簡潔に述べてください。\n\n"
            "判定:\n"
            "- 結果の適切性: タスクの実行結果は適切だったと思いますか?あなたの判断を真偽値で示してください。\n"
            "- 判定の自信度: 上記の判断に対するあなたの自信の度合いを0から1までの小数で示してください。\n"
            "- 判定の理由: タスクの実行結果の適切性とそれに対する自信度について、判断に至った理由を簡潔に列挙してください。\n\n"
            "出力は必ず日本語で行ってください。\n\n"
            "Tips: Make sure to answer in the correct format."
        )

        chain = prompt | self.llm

        @retry(tries=5)
        def invoke_chain() -> Reflection:
            return chain.invoke({"task": task, "result": result})

        reflection = invoke_chain()
        reflection_id = self.reflection_manager.save_reflection(reflection)
        reflection.id = reflection_id

        return reflection
