from typing import Generator

from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
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


class NaiveRAGChain(BaseRAGChain):
    def __init__(self, model: BaseChatModel):
        self.model = model

        # 検索の準備
        embeddings = init_embeddings(model="text-embedding-3-small", provider="openai")
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./tmp/chroma",
        )
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    @traceable(name="naive", reduce_fn=reduce_fn)
    def stream(self, question: str) -> Generator[Context | AnswerToken, None, None]:
        # 検索して検索結果を返す
        documents = self.retriever.invoke(question)
        yield Context(documents=documents)

        # 回答を生成して徐々に応答を返す
        prompt = _generate_answer_prompt_template.format(
            context=documents,
            question=question,
        )
        for chunk in self.model.stream(prompt):
            yield AnswerToken(token=chunk.content)


def create_naive_rag_chain(model: BaseChatModel) -> BaseRAGChain:
    return NaiveRAGChain(model)
