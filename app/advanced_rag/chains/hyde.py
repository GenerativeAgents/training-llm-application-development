from typing import Generator

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable

from app.advanced_rag.chains.base import AnswerToken, BaseRAGChain, Context

_hypothetical_prompt_template = """\
次の質問に回答する一文を書いてください。

質問: {question}
"""

_generate_answer_prompt_template = '''
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
'''


class HyDERAGChain(BaseRAGChain):
    def __init__(self, model: BaseChatModel):
        # 仮説的な回答を生成するChainの準備
        hypothetical_prompt = ChatPromptTemplate.from_template(
            _hypothetical_prompt_template
        )
        self.hypothetical_chain = hypothetical_prompt | model | StrOutputParser()

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

    @traceable(name="hyde")
    def stream(self, question: str) -> Generator[Context | AnswerToken, None, None]:
        # 仮説的な回答を生成する
        hypothetical_answer = self.hypothetical_chain.invoke({"question": question})

        # 検索して検索結果を返す
        documents = self.retriever.invoke(hypothetical_answer)
        yield Context(documents=documents)

        # 回答を生成して徐々に応答を返す
        for chunk in self.generate_answer_chain.stream(
            {"context": documents, "question": question}
        ):
            yield AnswerToken(token=chunk)


def create_hyde_rag_chain(model: BaseChatModel) -> BaseRAGChain:
    return HyDERAGChain(model)
