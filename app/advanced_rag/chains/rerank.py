from typing import Generator, Sequence

from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from app.advanced_rag.chains.base import AnswerToken, BaseRAGChain, Context

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
    cohere_reranker = CohereRerank(
        model="rerank-v3.5",
        top_n=top_n,
    )
    return cohere_reranker.compress_documents(documents=documents, query=question)


class RerankRAGChain(BaseRAGChain):
    def __init__(self, model: BaseChatModel):
        # 検索の準備
        embeddings = init_embeddings(model="text-embedding-3-small", provider="openai")
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./tmp/chroma",
        )
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 20})

        # 回答生成のChainの準備
        generate_answer_prompt = ChatPromptTemplate.from_template(
            _generate_answer_prompt_template
        )
        self.generate_answer_chain = generate_answer_prompt | model | StrOutputParser()

    @traceable(name="rerank")
    def stream(self, question: str) -> Generator[Context | AnswerToken, None, None]:
        # 検索する
        retrieved_documents = self.retriever.invoke(question)
        # リランクする
        documents = _rerank(question, retrieved_documents, top_n=5)
        # ドキュメントを返す
        yield Context(documents=documents)

        # 回答を生成して徐々に応答を返す
        for chunk in self.generate_answer_chain.stream(
            {"context": documents, "question": question}
        ):
            yield AnswerToken(token=chunk)


def create_rerank_rag_chain(model: BaseChatModel) -> BaseRAGChain:
    return RerankRAGChain(model)
