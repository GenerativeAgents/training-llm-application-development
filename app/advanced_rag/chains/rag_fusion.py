from typing import Generator

from langchain.load import dumps, loads
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable
from pydantic import BaseModel, Field

from app.advanced_rag.chains.base import AnswerToken, BaseRAGChain, Context


class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="検索クエリのリスト")


_query_generation_prompt_template = """\
質問に対してベクターデータベースから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似性検索の限界を克服するために、
ユーザーの質問に対して複数の視点を提供することが目標です。

質問: {question}
"""


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


class RAGFusionRAGChain(BaseRAGChain):
    def __init__(self, model: BaseChatModel):
        # 検索クエリを生成するChainの準備
        query_generation_prompt = ChatPromptTemplate.from_template(
            _query_generation_prompt_template
        )
        self.query_generation_chain: Runnable[dict[str, str], QueryGenerationOutput] = (
            query_generation_prompt
            | model.with_structured_output(QueryGenerationOutput)  # type: ignore[assignment]
        )

        # 検索の準備
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory="./tmp/chroma",
        )
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # 回答生成のChainの準備
        generate_answer_prompt = ChatPromptTemplate.from_template(
            _generate_answer_prompt_template
        )
        self.generate_answer_chain = generate_answer_prompt | model | StrOutputParser()

    @traceable(name="rag_fusion")
    def stream(self, question: str) -> Generator[Context | AnswerToken, None, None]:
        # 検索クエリを生成する
        query_generation_output = self.query_generation_chain.invoke(
            {"question": question}
        )

        # 検索する
        documents_list = self.retriever.batch(query_generation_output.queries)
        # 検索結果をRRFで融合する
        fused_documents = _reciprocal_rank_fusion(documents_list)
        # 上位5件のドキュメントに絞って返す
        documents = fused_documents[:5]
        yield Context(documents=documents)

        # 回答を生成して徐々に応答を返す
        for chunk in self.generate_answer_chain.stream(
            {"context": documents, "question": question}
        ):
            yield AnswerToken(token=chunk)


def create_rag_fusion_chain(model: BaseChatModel) -> BaseRAGChain:
    return RAGFusionRAGChain(model)
