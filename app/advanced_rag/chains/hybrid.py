from typing import Generator

from langchain.embeddings import init_embeddings
from langchain.load import dumps, loads
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
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


class HybridRAGChain(BaseRAGChain):
    def __init__(self, model: BaseChatModel):
        # Embeddingモデルを使ったベクトル検索の準備
        embeddings = init_embeddings(model="text-embedding-3-small", provider="openai")
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./tmp/chroma",
        )
        self.chroma_retriever = vector_store.as_retriever(
            search_kwargs={"k": 20}
        ).with_config({"run_name": "chroma_retriever"})

        # BM25を使った検索の準備
        loader = DirectoryLoader(
            path="tmp/langchain",
            glob="**/*.mdx",
            loader_cls=TextLoader,
        )
        documents = loader.load()
        self.bm25_retriever = BM25Retriever.from_documents(documents, k=10).with_config(
            {"run_name": "bm25_retriever"}
        )

        # 回答生成のChainの準備
        generate_answer_prompt = ChatPromptTemplate.from_template(
            _generate_answer_prompt_template
        )
        self.generate_answer_chain = generate_answer_prompt | model | StrOutputParser()

    @traceable(name="hybrid")
    def stream(self, question: str) -> Generator[Context | AnswerToken, None, None]:
        # 並列で検索する準備
        parallel_retriever = RunnableParallel(
            {
                "chroma_documents": self.chroma_retriever,
                "bm25_documents": self.bm25_retriever,
            }
        )
        # 並列で検索する
        parallel_retriever_output = parallel_retriever.invoke(question)
        chroma_documents = parallel_retriever_output["chroma_documents"]
        bm25_documents = parallel_retriever_output["bm25_documents"]

        # 検索結果をRRFで融合する
        fused_documents = _reciprocal_rank_fusion([chroma_documents, bm25_documents])
        # 上位5件のドキュメントに絞って返す
        documents = fused_documents[:5]
        yield Context(documents=documents)

        # 回答を生成して徐々に応答を返す
        for chunk in self.generate_answer_chain.stream(
            {"context": documents, "question": question}
        ):
            yield AnswerToken(token=chunk)


def create_hybrid_rag_chain(model: BaseChatModel) -> BaseRAGChain:
    return HybridRAGChain(model)
