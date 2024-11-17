from typing import Any

from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class Rerank:
    def __init__(self, top_n: int = 3):
        self.top_n = top_n

    def __call__(self, inputs: dict[str, Any]) -> list[Document]:
        question = inputs["question"]
        documents = inputs["documents"]

        cohere_reranker = CohereRerank(
            model="rerank-multilingual-v3.0",
            top_n=self.top_n,
        )
        return cohere_reranker.compress_documents(documents=documents, query=question)


_prompt_template = '''
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
'''


def create_rerank_rag_chain() -> Runnable[str, dict[str, Any]]:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory="./tmp/chroma",
    )

    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_template(_prompt_template)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    rerank = Rerank()

    return (
        RunnableParallel(
            {
                "documents": retriever,
                "question": RunnablePassthrough(),
            }
        ).with_types(input_type=str)
        | RunnablePassthrough.assign(context=rerank)
        | RunnablePassthrough.assign(answer=prompt | model | StrOutputParser())
    )
