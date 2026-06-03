from copy import deepcopy
from typing import Generator, Sequence

import boto3
import weave
from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from app.advanced_rag.chains.base import (
    AnswerToken,
    BaseRAGChain,
    Context,
    WeaveCallId,
    accumulator,
)

_generate_answer_prompt_template = '''
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
'''

_aws_region = "ap-northeast-1"
_rerank_model_arn = (
    f"arn:aws:bedrock:{_aws_region}::foundation-model/cohere.rerank-v3-5:0"
)


@weave.op
def _rerank(
    question: str, documents: Sequence[Document], top_n: int
) -> Sequence[Document]:
    documents_str = [doc.page_content for doc in documents]

    client = boto3.client("bedrock-agent-runtime", region_name=_aws_region)
    sources = [
        {
            "type": "INLINE",
            "inlineDocumentSource": {
                "type": "TEXT",
                "textDocument": {"text": doc_str},
            },
        }
        for doc_str in documents_str
    ]
    response = client.rerank(
        queries=[
            {
                "type": "TEXT",
                "textQuery": {"text": question},
            }
        ],
        sources=sources,
        rerankingConfiguration={
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {"modelArn": _rerank_model_arn},
                "numberOfResults": top_n,
            },
        },
    )

    reranked_documents: list[Document] = []
    for result in response["results"]:
        index = result["index"]
        relevance_score = result["relevanceScore"]

        doc = documents[index]
        doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
        doc_copy.metadata["relevance_score"] = relevance_score
        reranked_documents.append(doc_copy)

    return reranked_documents


class RerankRAGChain(BaseRAGChain):
    def __init__(self, model: BaseChatModel):
        self.model = model

        # 検索の準備
        embeddings = init_embeddings(model="text-embedding-3-small", provider="openai")
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./tmp/chroma",
        )
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 20})

    @weave.op(name="rerank", accumulator=accumulator)
    def stream(
        self, question: str
    ) -> Generator[Context | AnswerToken | WeaveCallId, None, None]:
        current_call = weave.require_current_call()
        yield WeaveCallId(weave_call_id=current_call.id)

        # 検索する
        retrieved_documents = self.retriever.invoke(question)
        # リランクする
        documents = _rerank(question, retrieved_documents, top_n=5)
        # ドキュメントを返す
        yield Context(documents=documents)

        # 回答を生成して徐々に応答を返す
        prompt = _generate_answer_prompt_template.format(
            context=documents,
            question=question,
        )
        for chunk in self.model.stream(prompt):
            yield AnswerToken(token=chunk.content)


def create_rerank_rag_chain(model: BaseChatModel) -> BaseRAGChain:
    return RerankRAGChain(model)
