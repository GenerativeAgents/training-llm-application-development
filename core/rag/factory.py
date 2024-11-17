from enum import Enum
from typing import Any

from langchain_core.runnables import Runnable

from core.rag.hyde import create_hyde_rag_chain
from core.rag.multi_query import create_multi_query_rag_chain
from core.rag.naive import create_naive_rag_chain
from core.rag.rag_fusion import create_rag_fusion_chain


class RAGChainType(str, Enum):
    NAIVE = "naive"
    HYDE = "hyde"
    MULTI_QUERY = "multi_query"
    RAG_FUSION = "fag_fusion"


def create_rag_chain(rag_chain_type: RAGChainType) -> Runnable[str, dict[str, Any]]:
    if rag_chain_type == RAGChainType.NAIVE:
        return create_naive_rag_chain()
    elif rag_chain_type == RAGChainType.HYDE:
        return create_hyde_rag_chain()
    elif rag_chain_type == RAGChainType.MULTI_QUERY:
        return create_multi_query_rag_chain()
    elif rag_chain_type == RAGChainType.RAG_FUSION:
        return create_rag_fusion_chain()
    else:
        raise ValueError(f"Unknown RAG chain type: {rag_chain_type}")
