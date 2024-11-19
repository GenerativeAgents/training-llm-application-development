from enum import Enum
from typing import Any

from langchain_core.runnables import Runnable

from app.rag.chains.hybrid import create_hybrid_rag_chain
from app.rag.chains.hyde import create_hyde_rag_chain
from app.rag.chains.multi_query import create_multi_query_rag_chain
from app.rag.chains.naive import create_naive_rag_chain
from app.rag.chains.rag_fusion import create_rag_fusion_chain
from app.rag.chains.rerank import create_rerank_rag_chain
from app.rag.chains.route import create_route_rag_chain


class RAGChainType(str, Enum):
    NAIVE = "naive"
    HYDE = "hyde"
    MULTI_QUERY = "multi_query"
    RAG_FUSION = "fag_fusion"
    RERANK = "rerank"
    ROUTE = "route"
    HYBRID = "hybrid"


def create_rag_chain(rag_chain_type: RAGChainType) -> Runnable[str, dict[str, Any]]:
    if rag_chain_type == RAGChainType.NAIVE:
        chain = create_naive_rag_chain()
    elif rag_chain_type == RAGChainType.HYDE:
        chain = create_hyde_rag_chain()
    elif rag_chain_type == RAGChainType.MULTI_QUERY:
        chain = create_multi_query_rag_chain()
    elif rag_chain_type == RAGChainType.RAG_FUSION:
        chain = create_rag_fusion_chain()
    elif rag_chain_type == RAGChainType.RERANK:
        chain = create_rerank_rag_chain()
    elif rag_chain_type == RAGChainType.ROUTE:
        chain = create_route_rag_chain()
    elif rag_chain_type == RAGChainType.HYBRID:
        chain = create_hybrid_rag_chain()
    else:
        raise ValueError(f"Unknown RAG chain type: {rag_chain_type}")

    return chain.with_config({"run_name": rag_chain_type})