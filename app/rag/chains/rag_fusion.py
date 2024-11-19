from typing import Any

from langchain.load import dumps, loads
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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


def reciprocal_rank_fusion(
    retriever_outputs: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    # 各ドキュメントの文字列とそのスコアの対応を保持する辞書を準備
    content_score_mapping: dict[str, float] = {}

    # 検索クエリごとにループ
    for docs in retriever_outputs:
        # 検索結果のドキュメントごとにループ
        for rank, doc in enumerate(docs):
            # ドキュメントをメタデータ含め文字列化
            doc_str = dumps(doc)

            # 初めて登場したコンテンツの場合はスコアを0で初期化
            if doc_str not in content_score_mapping:
                content_score_mapping[doc_str] = 0

            # (1 / (順位 + k)) のスコアを加算
            content_score_mapping[doc_str] += 1 / (rank + k)

    # スコアの大きい順にソート
    ranked = sorted(content_score_mapping.items(), key=lambda x: x[1], reverse=True)  # noqa
    return [loads(doc_str) for doc_str, _ in ranked]


def create_rag_fusion_chain() -> Runnable[str, dict[str, Any]]:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory="./tmp/chroma",
    )

    retriever = vector_store.as_retriever()
    rag_prompt = ChatPromptTemplate.from_template(_rag_prompt_template)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    query_generation_chain: Runnable[str, list[str]] = (
        ChatPromptTemplate.from_template(_query_generation_prompt)
        | model.with_structured_output(QueryGenerationOutput)
        | (lambda x: x.queries)  # type: ignore[assignment]
    )

    return RunnableParallel(
        {
            "context": query_generation_chain
            | retriever.map()
            | reciprocal_rank_fusion,
            "question": RunnablePassthrough(),
        }
    ).with_types(input_type=str) | RunnablePassthrough.assign(
        answer=rag_prompt | model | StrOutputParser()
    )
