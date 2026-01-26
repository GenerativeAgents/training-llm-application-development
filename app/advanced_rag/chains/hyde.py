from typing import Generator

import weave
from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel

from app.advanced_rag.chains.base import AnswerToken, BaseRAGChain, Context, WeaveCallId
from app.prompts import generate_answer_prompt, hypothetical_prompt


class HyDERAGChain(BaseRAGChain):
    def __init__(self, model: BaseChatModel):
        self.model = model

        # 検索の準備
        embeddings = init_embeddings(model="text-embedding-3-small", provider="openai")
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./tmp/chroma",
        )
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    @weave.op(name="hyde")
    def stream(
        self, question: str
    ) -> Generator[Context | AnswerToken | WeaveCallId, None, None]:
        current_call = weave.require_current_call()
        yield WeaveCallId(weave_call_id=current_call.id)

        # 仮説的な回答を生成
        hypothetical_prompt_text = hypothetical_prompt.format(question=question)
        hypothetical_answer = self.model.invoke(hypothetical_prompt_text)

        # 検索して検索結果を返す
        documents = self.retriever.invoke(hypothetical_answer.content)
        yield Context(documents=documents)

        # 回答を生成して徐々に応答を返す
        generate_answer_prompt_text = generate_answer_prompt.format(
            context=documents,
            question=question,
        )
        for chunk in self.model.stream(generate_answer_prompt_text):
            yield AnswerToken(token=chunk.content)


def create_hyde_rag_chain(model: BaseChatModel) -> BaseRAGChain:
    return HyDERAGChain(model)
