from enum import Enum
from typing import Generator

from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langsmith import traceable
from pydantic import BaseModel

from app.advanced_rag.chains.base import AnswerToken, BaseRAGChain, Context


class Route(str, Enum):
    langchain_document = "langchain_document"
    web = "web"


class RouteOutput(BaseModel):
    route: Route


_route_prompt_template = """\
質問に回答するために適切なRetrieverを選択してください。
用意しているのは、LangChainに関する情報を検索する「langchain_document」と、
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
        # ルーティングのChainの準備
        route_prompt = ChatPromptTemplate.from_template(_route_prompt_template)
        self.route_chain: Runnable[dict[str, str], RouteOutput] = (
            route_prompt | model.with_structured_output(RouteOutput)  # type: ignore[assignment]
        )

        # LangChainのドキュメントを検索する準備
        embeddings = init_embeddings(model="text-embedding-3-small", provider="openai")
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./tmp/chroma",
        )
        self.langchain_document_retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        ).with_config({"run_name": "langchain_document_retriever"})

        # Web検索の準備
        self.web_retriever = TavilySearchAPIRetriever(k=5).with_config(
            {"run_name": "web_retriever"}
        )

        # 回答生成のChainの準備
        generate_answer_prompt = ChatPromptTemplate.from_template(
            _generate_answer_prompt_template
        )
        self.generate_answer_chain = generate_answer_prompt | model | StrOutputParser()

    @traceable(name="route")
    def stream(self, question: str) -> Generator[Context | AnswerToken, None, None]:
        # ルーティング
        route_output = self.route_chain.invoke({"question": question})
        route = route_output.route

        # ルーティングに応じて検索
        if route == Route.langchain_document:
            documents = self.langchain_document_retriever.invoke(question)
        elif route == Route.web:
            documents = self.web_retriever.invoke(question)

        # 検索結果を返す
        yield Context(documents=documents)

        # 回答を生成して徐々に応答を返す
        for chunk in self.generate_answer_chain.stream(
            {"context": documents, "question": question}
        ):
            yield AnswerToken(token=chunk)


def create_route_rag_chain(model: BaseChatModel) -> BaseRAGChain:
    return RouteRAGChain(model)
