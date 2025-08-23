from abc import ABC, abstractmethod
from typing import Generator, Sequence

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
