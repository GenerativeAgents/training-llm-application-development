from typing import Any

from langchain.load import dumps, loads
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings


def _reciprocal_rank_fusion(
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


def _create_hybrid_retriever() -> RetrieverLike:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory="./tmp/chroma",
    )

    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 10}).with_config(
        {"run_name": "chroma_retriever"}
    )

    loader = DirectoryLoader(
        # ../tmp/langchain ではないので注意
        path="tmp/langchain",
        glob="**/*.mdx",
        loader_cls=TextLoader,
    )
    documents = loader.load()
    bm25_retriever = BM25Retriever.from_documents(documents, k=10).with_config(
        {"run_name": "bm25_retriever"}
    )

    return (
        RunnableParallel(
            {
                "chroma_documents": chroma_retriever,
                "bm25_documents": bm25_retriever,
            }
        ).with_types(input_type=str)
        | (lambda x: [x["chroma_documents"], x["bm25_documents"]])
        | _reciprocal_rank_fusion
        | (lambda x: x[:5])  # 上位5件のドキュメントを取得
    )


_prompt_template = '''
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
'''


def create_hybrid_rag_chain(model: BaseChatModel) -> Runnable[str, dict[str, Any]]:
    retriever = _create_hybrid_retriever()

    prompt = ChatPromptTemplate.from_template(_prompt_template)

    return RunnableParallel(
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
    ).with_types(input_type=str) | RunnablePassthrough.assign(
        answer=prompt | model | StrOutputParser()
    )
