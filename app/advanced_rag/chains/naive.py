from typing import Generator

from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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

    @traceable(name="naive", reduce_fn=reduce_fn)
    def stream(self, question: str) -> Generator[Context | AnswerToken, None, None]:
        # 検索して検索結果を返す
        documents = self.retriever.invoke(question)
        yield Context(documents=documents)

        # 回答を生成して徐々に応答を返す
        for chunk in self.generate_answer_chain.stream(
            {"context": documents, "question": question}
        ):
            yield AnswerToken(token=chunk)


def create_naive_rag_chain(model: BaseChatModel) -> BaseRAGChain:
    return NaiveRAGChain(model)
