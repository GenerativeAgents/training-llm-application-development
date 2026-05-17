from copy import deepcopy
from typing import Generator, Sequence

import cohere
from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langsmith import traceable

from app.advanced_rag.chains.base import AnswerToken, BaseRAGChain, Context, reduce_fn

_generate_answer_prompt_template = '''
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
'''


@traceable
def _rerank(
    question: str, documents: Sequence[Document], top_n: int
) -> Sequence[Document]:
    documents_str = [doc.page_content for doc in documents]

    client = cohere.ClientV2()
    response = client.rerank(
        model="rerank-v3.5",
        query=question,
        documents=documents_str,
        top_n=top_n,
    )

    reranked_documents: list[Document] = []
    for result in response.results:
        doc = documents[result.index]
        doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
        doc_copy.metadata["relevance_score"] = result.relevance_score
        reranked_documents.append(doc_copy)

    return reranked_documents


class RerankRAGChain(BaseRAGChain):
    def __init__(self, model: BaseChatModel):
        self.model = model

        # 検索の準備
        embeddings = init_embeddings(model="text-embedding-3-small", provider="openai")
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./tmp/chroma",
        )
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 20})

    @traceable(name="rerank", reduce_fn=reduce_fn)
    def stream(self, question: str) -> Generator[Context | AnswerToken, None, None]:
        # 検索する
        retrieved_documents = self.retriever.invoke(question)
        # リランクする
        documents = _rerank(question, retrieved_documents, top_n=5)
        # ドキュメントを返す
        yield Context(documents=documents)

        # 回答を生成して徐々に応答を返す
        prompt = _generate_answer_prompt_template.format(
            context=documents,
            question=question,
        )
        for chunk in self.model.stream(prompt):
            yield AnswerToken(token=chunk.content)


def create_rerank_rag_chain(model: BaseChatModel) -> BaseRAGChain:
    return RerankRAGChain(model)
