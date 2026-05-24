# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational repository for an LLM application development course (LLM アプリケーション開発者養成講座). Contains source code, Jupyter notebooks, and Streamlit apps demonstrating RAG, agents, and LangGraph patterns. Documentation and comments are in Japanese.

## Commands

```bash
# Install dependencies
uv sync

# Run Streamlit web app (port 8080)
make streamlit
# or: uv run streamlit run app.py --server.port 8080

# Run Jupyter notebooks
make jupyter

# Run all notebooks as tests (executes every .ipynb in notebooks/)
make test

# Clear notebook outputs
make clean
```

## Architecture

### Advanced RAG (`app/advanced_rag/`)
Factory-pattern RAG system with pluggable retrieval strategies. All chains extend `BaseRAGChain` (in `chains/base.py`) and implement a `stream()` method that yields `Context` (retrieved documents), `AnswerToken` (streaming answer tokens), and `WeaveCallId` (Weave tracing). The factory in `factory.py` maps chain names to constructors:
- `naive` - basic retrieve-and-generate
- `hyde` - Hypothetical Document Embeddings
- `multi_query` - multiple query variations
- `rag_fusion` - fused results from multiple queries
- `rerank` - Cohere-based reranking
- `route` - dynamic routing between retrievers
- `hybrid` - combined BM25 + semantic search

### MCP Server (`app/random_number_mcp.py`)
Example MCP (Model Context Protocol) server used by the Streamlit MCP pages.

### Streamlit Pages (`pages/`)
Progressive examples organized by course part. Each file is a standalone Streamlit page:
- **part1** - Chatbot, workflow, agent, MCP, checkpointer, human-in-the-loop, DeepAgents
- **part2** - Indexing, RAG, advanced RAG
- **part3** - Dataset creation, evaluation, advanced RAG with feedback
- **partX** - Supervisor agent pattern

The main entry point is `app.py` (simple chatbot).

### Notebooks (`notebooks/`)
Jupyter notebooks for interactive teaching. Executed as tests via `make test`.

## Key Technical Details

- **Python 3.11**, managed with **uv** (dependencies in `pyproject.toml`, lock in `uv.lock`)
- **LangChain** + **LangGraph** for chains and agent orchestration
- **Streamlit** for web UI with `st.write_stream()` for streaming responses
- **Chroma** vector store persisted at `./tmp/chroma`, using OpenAI `text-embedding-3-small` embeddings
- **Weave** (Weights & Biases) for tracing and evaluation
- Environment variables loaded from `.env` (see `.env.template`): `OPENAI_API_KEY`, `WANDB_API_KEY`, `COHERE_API_KEY`, `TAVILY_API_KEY`
