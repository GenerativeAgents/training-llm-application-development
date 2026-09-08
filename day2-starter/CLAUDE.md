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
Example MCP (Model Context Protocol) server used by the MCP notebook (`notebooks/part1_4_mcp.ipynb`) and the parked Streamlit MCP pages.

### Streamlit Pages (`pages/`)
Progressive examples organized by course part. Each file is a standalone Streamlit page:
- **part1** - Chatbot (`part1_1`), workflow (`part1_2`), agent with tools (`part1_3`)
- **part2** - Advanced RAG
- **part3** - Dataset creation, evaluation, advanced RAG with feedback
- **partX** - Not used in the current course flow, kept as references: human-in-the-loop (`partX_1`), DeepAgents (`partX_2`), supervisor (`partX_3`), MCP via LangChain (`partX_4`, `partX_5`)

The main entry point is `app.py` (simple chatbot).

### Notebooks (`notebooks/`)
Jupyter notebooks for interactive teaching. Executed as tests via `make test`.
- `part1_1_llm_api_basics` - Chat Completions API, Vision, reasoning_effort, LangChain Model
- `part1_2_workflow_and_agent` - Structured outputs, LangGraph workflow, Function calling, agent loop, `create_agent`
- `part1_3_coding_agent` - Minimal coding agent (run_command / read_file / write_file + agent loop), working dir `tmp/coding-agent`
- `part1_4_mcp` - MCP servers (DeepWiki, Serena, custom) called from an async agent loop with the `mcp` SDK, no LangChain
- `part2_1_rag_basics` - RAG basics with Chroma and Weave
- `partX_1_langgraph_basics` - LangGraph basics (parked, not in the current course flow)

Notes on models: all code uses `gpt-5.6-luna`. With the Chat Completions API, GPT-5.6 / GPT-6 accept function tools only with `reasoning_effort="none"`.

## Key Technical Details

- **Python 3.13**, managed with **uv** (dependencies in `pyproject.toml`, lock in `uv.lock`)
- **LangChain** + **LangGraph** for chains and agent orchestration
- **Streamlit** for web UI with `st.write_stream()` for streaming responses
- **Chroma** vector store persisted at `./tmp/chroma`, using OpenAI `text-embedding-3-small` embeddings
- **Weave** (Weights & Biases) for tracing and evaluation
- Environment variables loaded from `.env` (see `.env.template`): `OPENAI_API_KEY`, `WANDB_API_KEY`, `WANDB_PROJECT`
- **Web search** via Amazon Bedrock AgentCore Gateway Web Search Tool (`app/tools/web_search.py`): exposes a plain `web_search()` function with no LangChain dependency; callers wrap it themselves (`@tool` in pages/notebooks, a custom `BaseRetriever` in `advanced_rag/chains/route.py`). Reads the gateway URL from `AGENTCORE_GATEWAY_URL` and the tool name from `AGENTCORE_WEB_SEARCH_TOOL_NAME` (both set by the hands-on environment; the SigV4 region is derived from the URL) and signs requests with AWS credentials (EC2 instance role); no API key needed
