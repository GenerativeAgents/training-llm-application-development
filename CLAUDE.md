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

# Run all notebooks as tests
make test

# Clear notebook outputs
make clean

# Run documentation agent CLI
uv run python -m app.documentation_agent.agent --task "description" --k 5

# Run CLI chatbot
uv run python app/chat_cli.py
```

## Architecture

### Advanced RAG (`app/advanced_rag/`)
Factory-pattern RAG system with pluggable retrieval strategies. All chains extend `BaseRAGChain` (in `chains/base.py`) and implement a `stream()` method that yields `Context` (retrieved documents) and `AnswerToken` (streaming answer tokens). The factory in `factory.py` maps chain names to constructors:
- `naive` - basic retrieve-and-generate
- `hyde` - Hypothetical Document Embeddings
- `multi_query` - multiple query variations
- `rag_fusion` - fused results from multiple queries
- `rerank` - Cohere-based reranking
- `route` - dynamic routing between retrievers
- `hybrid` - combined BM25 + semantic search

### Agent Design Patterns (`app/agent_design_pattern/`)
LangGraph-based agent implementations demonstrating different patterns: prompt optimization, response optimization, self-reflection, role-based cooperation, single-path plan generation, and passive goal creation. Each pattern lives in its own subdirectory with a `main.py`. Shared config is in `settings.py` using `pydantic-settings` (loads from `.env`). A common `reflection_manager.py` provides self-evaluation capabilities.

### Documentation Agent (`app/documentation_agent/agent.py`)
Multi-step LangGraph agent that generates requirements documents through iterative persona-based interviews: generates personas → conducts interviews → evaluates information sufficiency → loops or generates final document. Uses Pydantic models for structured LLM outputs and state management.

### Streamlit Pages (`pages/`)
Progressive examples organized by course day (day1–day3). Each file is a standalone Streamlit page covering topics from basic indexing to supervisor agent patterns. The main entry point is `app.py` (simple chatbot).

### Notebooks (`notebooks/`)
Jupyter notebooks for interactive teaching, also organized by day. Executed as tests via `make test`.

## Key Technical Details

- **Python 3.11**, managed with **uv** (dependencies in `pyproject.toml`, lock in `uv.lock`)
- **LangChain 1.0.x** + **LangGraph 1.0.x** for chains and agent orchestration
- **Streamlit** for web UI with `st.write_stream()` for streaming responses
- **Chroma** vector store persisted at `./tmp/chroma`, using OpenAI `text-embedding-3-small` embeddings
- **Pydantic models** used extensively for structured LLM outputs, agent state, and configuration
- Environment variables loaded from `.env` (see `.env.template`): `OPENAI_API_KEY` (required), plus optional `LANGCHAIN_API_KEY`, `COHERE_API_KEY`, `TAVILY_API_KEY`
- LangSmith tracing via `@traceable` decorators and `LANGCHAIN_TRACING_V2=true`
- Code formatted with **Ruff**, type-checked with **Mypy**
- Default LLM models: `gpt-5-nano` (app.py chatbot), `gpt-4.1` (agents and RAG)
