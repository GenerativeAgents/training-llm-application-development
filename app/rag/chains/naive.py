from typing import Any

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

_prompt_template = '''
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
'''


def create_naive_rag_chain(model: BaseChatModel) -> Runnable[str, dict[str, Any]]:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory="./tmp/chroma",
    )

    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_template(_prompt_template)

    return RunnableParallel(
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
    ).with_types(input_type=str) | RunnablePassthrough.assign(
        answer=prompt | model | StrOutputParser()
    )
