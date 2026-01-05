from enum import Enum
from typing import Generator

import weave
from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from app.advanced_rag.chains.base import AnswerToken, BaseRAGChain, Context, WeaveCallId


class Route(str, Enum):
    langsmith_document = "langsmith_document"
    web = "web"


class RouteOutput(BaseModel):
    route: Route


_route_prompt_template = """\
質問に回答するために適切なRetrieverを選択してください。
用意しているのは、LangSmithに関する情報を検索する「langsmith_document」と、
それ以外の質問をWebサイトで検索するための「web」です。

質問: {question}
"""


_generate_answer_prompt_template = '''
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
'''


class RouteRAGChain(BaseRAGChain):
    def __init__(self, model: BaseChatModel):
        self.model = model
        # LangChainのドキュメントを検索する準備
        embeddings = init_embeddings(model="text-embedding-3-small", provider="openai")
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./tmp/chroma",
        )
        self.langsmith_document_retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        ).with_config({"run_name": "langsmith_document_retriever"})

        # Web検索の準備
        self.web_retriever = TavilySearchAPIRetriever(k=5).with_config(
            {"run_name": "web_retriever"}
        )

    @weave.op(name="route")
    def stream(
        self, question: str
    ) -> Generator[Context | AnswerToken | WeaveCallId, None, None]:
        current_call = weave.require_current_call()
        yield WeaveCallId(weave_call_id=current_call.id)

        # ルーティング
        route_prompt = _route_prompt_template.format(question=question)
        model_with_structure = self.model.with_structured_output(RouteOutput)
        route_output: RouteOutput = model_with_structure.invoke(route_prompt)  # type: ignore[assignment]
        route = route_output.route

        # ルーティングに応じて検索
        if route == Route.langsmith_document:
            documents = self.langsmith_document_retriever.invoke(question)
        elif route == Route.web:
            documents = self.web_retriever.invoke(question)

        # 検索結果を返す
        yield Context(documents=documents)

        # 回答を生成して徐々に応答を返す
        generate_answer_prompt = _generate_answer_prompt_template.format(
            context=documents,
            question=question,
        )
        for chunk in self.model.stream(generate_answer_prompt):
            yield AnswerToken(token=chunk.content)


def create_route_rag_chain(model: BaseChatModel) -> BaseRAGChain:
    return RouteRAGChain(model)
