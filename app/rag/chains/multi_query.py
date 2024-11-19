from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field


class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="検索クエリのリスト")


_query_generation_prompt = """\
質問に対してベクターデータベースから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似性検索の限界を克服するために、
ユーザーの質問に対して複数の視点を提供することが目標です。

質問: {question}
"""


_rag_prompt_template = '''
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
'''


def flatten(nested_list: list[list[Document]]) -> list[Document]:
    res = []
    for sublist in nested_list:
        res.extend(sublist)
    return res


def create_multi_query_rag_chain(model: BaseChatModel) -> Runnable[str, dict[str, Any]]:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory="./tmp/chroma",
    )

    retriever = vector_store.as_retriever()
    rag_prompt = ChatPromptTemplate.from_template(_rag_prompt_template)

    query_generation_chain: Runnable[str, list[str]] = (
        ChatPromptTemplate.from_template(_query_generation_prompt)
        | model.with_structured_output(QueryGenerationOutput)
        | (lambda x: x.queries)  # type: ignore[assignment]
    )

    return RunnableParallel(
        {
            "context": query_generation_chain | retriever.map() | flatten,
            "question": RunnablePassthrough(),
        }
    ).with_types(input_type=str) | RunnablePassthrough.assign(
        answer=rag_prompt | model | StrOutputParser()
    )
