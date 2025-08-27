from typing import Generator

from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langsmith import traceable
from pydantic import BaseModel, Field

from app.advanced_rag.chains.base import AnswerToken, BaseRAGChain, Context, reduce_fn


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


class MultiQueryRAGChain(BaseRAGChain):
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
        embeddings = init_embeddings(model="text-embedding-3-small", provider="openai")
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./tmp/chroma",
        )
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # 回答生成のChainの準備
        generate_answer_prompt = ChatPromptTemplate.from_template(
            _generate_answer_prompt_template
        )
        self.generate_answer_chain = generate_answer_prompt | model | StrOutputParser()

    @traceable(name="multi_query", reduce_fn=reduce_fn)
    def stream(self, question: str) -> Generator[Context | AnswerToken, None, None]:
        # 検索クエリを生成する
        query_generation_output = self.query_generation_chain.invoke(
            {"question": question}
        )

        # 検索して検索結果を返す
        documents_list = self.retriever.batch(query_generation_output.queries)
        documents = [doc for docs in documents_list for doc in docs]
        yield Context(documents=documents)

        # 回答を生成して徐々に応答を返す
        for chunk in self.generate_answer_chain.stream(
            {"context": documents, "question": question}
        ):
            yield AnswerToken(token=chunk)


def create_multi_query_rag_chain(model: BaseChatModel) -> BaseRAGChain:
    return MultiQueryRAGChain(model)
