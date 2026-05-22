from abc import ABC, abstractmethod
from typing import Generator, Sequence

from langchain_core.documents import Document


class Context:
    def __init__(self, documents: Sequence[Document]):
        self.documents = documents


class AnswerToken:
    def __init__(self, token: str):
        self.token = token


class WeaveCallId:
    def __init__(self, weave_call_id: str | None):
        self.weave_call_id = weave_call_id


class BaseRAGChain(ABC):
    @abstractmethod
    def stream(
        self, question: str
    ) -> Generator[Context | AnswerToken | WeaveCallId, None, None]:
        pass


def accumulator(acc: str | None, val: Context | AnswerToken | WeaveCallId) -> str:
    if acc is None:
        acc = ""

    if isinstance(val, AnswerToken):
        acc += val.token

    return acc
