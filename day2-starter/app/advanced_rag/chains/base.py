from abc import ABC, abstractmethod
from typing import Any, Generator, Sequence

from langchain_core.documents import Document


class Context:
    def __init__(self, documents: Sequence[Document]):
        self.documents = documents


class AnswerToken:
    def __init__(self, token: str):
        self.token = token


class BaseRAGChain(ABC):
    @abstractmethod
    def stream(self, question: str) -> Generator[Context | AnswerToken, None, None]:
        pass


def reduce_fn(chunks: Sequence[Context | AnswerToken]) -> Any:
    context: Sequence[Document] = []
    answer: str = ""

    for chunk in chunks:
        if isinstance(chunk, Context):
            context = chunk.documents

        if isinstance(chunk, AnswerToken):
            answer += chunk.token

    return {
        "context": context,
        "answer": answer,
    }
