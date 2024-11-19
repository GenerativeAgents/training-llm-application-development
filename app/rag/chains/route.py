from enum import Enum
from typing import Any

from langchain_chroma import Chroma
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel

_route_prompt = """\
質問に回答するために適切なRetrieverを選択してください。

質問: {question}
"""


def routed_retriever(inputs: dict[str, Any]) -> list[Document]:
    question = inputs["question"]
    route = inputs["route"]

    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory="./tmp/chroma",
    )
    langchain_document_retriever = vector_store.as_retriever().with_config(
        {"run_name": "langchain_document_retriever"}
    )

    web_retriever = TavilySearchAPIRetriever(k=3).with_config(
        {"run_name": "web_retriever"}
    )

    if route == Route.langchain_document:
        return langchain_document_retriever.invoke(question)
    elif route == Route.web:
        return web_retriever.invoke(question)

    raise ValueError(f"Unknown route: {route}")


class Route(str, Enum):
    langchain_document = "langchain_document"
    web = "web"


class RouteOutput(BaseModel):
    route: Route


_prompt_template = '''
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
'''


def create_route_rag_chain(model: BaseChatModel) -> Runnable[str, dict[str, Any]]:
    prompt = ChatPromptTemplate.from_template(_prompt_template)

    route_chain: Runnable[str, Route] = (
        ChatPromptTemplate.from_template(_route_prompt)
        | model.with_structured_output(RouteOutput)
        | (lambda x: x.route)  # type: ignore[assignment]
    )

    return (
        RunnableParallel(
            {
                "question": RunnablePassthrough(),
                "route": route_chain,
            }
        ).with_types(input_type=str)
        | RunnablePassthrough.assign(context=routed_retriever)
        | RunnablePassthrough.assign(answer=prompt | model | StrOutputParser())
    )
