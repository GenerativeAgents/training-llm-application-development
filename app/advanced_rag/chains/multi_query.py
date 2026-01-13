from typing import Generator

import weave
from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from app.advanced_rag.chains.base import AnswerToken, BaseRAGChain, Context, WeaveCallId


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
        self.model = model

        # 検索の準備
        embeddings = init_embeddings(model="text-embedding-3-small", provider="openai")
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./tmp/chroma",
        )
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    @weave.op(name="multi_query")
    def stream(
        self, question: str
    ) -> Generator[Context | AnswerToken | WeaveCallId, None, None]:
        current_call = weave.require_current_call()
        yield WeaveCallId(weave_call_id=current_call.id)

        # 検索クエリを生成する
        query_generation_prompt = _query_generation_prompt_template.format(
            question=question
        )
        model_with_structure = self.model.with_structured_output(QueryGenerationOutput)
        query_generation_output: QueryGenerationOutput = model_with_structure.invoke(
            query_generation_prompt
        )  # type: ignore[assignment]

        # 検索して検索結果を返す
        documents_list = self.retriever.batch(query_generation_output.queries)
        documents = [doc for docs in documents_list for doc in docs]
        yield Context(documents=documents)

        # 回答を生成して徐々に応答を返す
        generate_answer_prompt = _generate_answer_prompt_template.format(
            context=documents,
            question=question,
        )
        for chunk in self.model.stream(generate_answer_prompt):
            yield AnswerToken(token=chunk.content)


def create_multi_query_rag_chain(model: BaseChatModel) -> BaseRAGChain:
    return MultiQueryRAGChain(model)
