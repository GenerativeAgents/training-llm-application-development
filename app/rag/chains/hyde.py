from typing import Any

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

_hypothetical_prompt_template = """\
次の質問に回答する一文を書いてください。

質問: {question}
"""

_rag_prompt_template = '''
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
'''


def create_hyde_rag_chain(model: BaseChatModel) -> Runnable[str, dict[str, Any]]:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory="./tmp/chroma",
    )

    retriever = vector_store.as_retriever()
    rag_prompt = ChatPromptTemplate.from_template(_rag_prompt_template)

    hypothetical_chain: Runnable[str, str] = (
        ChatPromptTemplate.from_template(_hypothetical_prompt_template)
        | model
        | StrOutputParser()  # type: ignore[assignment]
    )

    return RunnableParallel(
        {
            "context": hypothetical_chain | retriever,
            "question": RunnablePassthrough(),
        }
    ).with_types(input_type=str) | RunnablePassthrough.assign(
        answer=rag_prompt | model | StrOutputParser()
    )
