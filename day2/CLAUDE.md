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

# Run the coding agent CLI
make coding-agent
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
Document ingestion, retrieval, answer generation, and evaluation follow this pipeline:
- `extractors/` - extract and chunk Excel, PDF, and PowerPoint documents using `basic`, `structured`, or `vision` extraction
- `ingest.py` - assign IDs and metadata to chunks from `data/corpus/`, reject duplicate IDs, and write `data/lancedb/docs_<extractor>.jsonl`
- `build_index.py` - run ingestion, embed chunks, and build a LanceDB table and Japanese full-text indexes for each extractor
- `rag.py` - `RagModel` retrieves with vector, full-text, or hybrid (RRF) search; optionally filters by collection, reranks with an LLM, selects and expands chunks, and generates an answer. `retrieve()` returns contexts and ranked IDs; `predict()` returns the answer and retrieval results, traced with Weave
- `eval.py` - evaluate retrieval and answer correctness with Weave datasets and `Evaluation`

### Agent loop (`app/agent_loop.py`) and coding agent CLI (`app/coding_agent.py`)
`agent_loop(messages, tools)` drives the Chat Completions API Function calling loop (same structure as the notebook version in `part1_2`, but yields each appended message so callers can display progress). `tools` is a list of `Tool` (API definition + callable); build each with `function_to_tool(func)`, which derives the JSON schema from the function's type hints and docstring (via Pydantic `create_model`), so there is no hand-written schema or name-to-function dict. `app/coding_agent.py` is a minimal coding agent CLI built on it with three tools (`run_command`, `read_file`, `write_file`) confined to `tmp/coding-agent`; run with `make coding-agent` or `uv run python -m app.coding_agent [--work-dir DIR]`.

### MCP Server (`app/random_number_mcp.py`)
Example MCP (Model Context Protocol) server shown in the slides; used by the parked Streamlit MCP pages.

### Streamlit Pages (`pages/`)
Progressive examples organized by course part. Each file is a standalone Streamlit page:
- **part1** - Chatbot (`part1_1`), workflow (`part1_2`), agent with tools on `app.agent_loop` (`part1_3`: web search, `run_command`, light switch)
- **part2** - Simple RAG with LanceDB and Weave feedback (`part2_1_rag`)
- **part3** - Advanced RAG with eight configurable stages (`part3_1_advanced_rag`)
- **partX** - Not used in the current course flow, kept as references: human-in-the-loop (`partX_1`), DeepAgents (`partX_2`), supervisor (`partX_3`), MCP via LangChain (`partX_4`, `partX_5`), the `create_agent` version of the agent page (`partX_6`)

The main entry point is `app.py` (simple chatbot).

`st.session_state` is shared across pages, so every page that keeps state calls `reset_session_state_on_page_change(__file__)` (from `app/session_state.py`) as the first line of `app()`; it clears the state when the page differs from the previous run. This lets all pages use the same keys (e.g. `st.session_state.messages`) even though part1_1 stores LangChain messages and part1_3 stores OpenAI dicts. Add the same line to any new page that uses `st.session_state`.

### Notebooks (`notebooks/`)
Jupyter notebooks for interactive teaching. Executed as tests via `make test`.
- `part1_1_llm_api_basics` - Chat Completions API, Vision, reasoning_effort, LangChain Model
- `part1_2_workflow_and_agent` - Structured outputs, LangGraph workflow, Function calling, agent loop, `create_agent`
- `part2_1_rag_basics` - RAG basics with LanceDB and Weave
- `part2_2_eval_basics` - Retrieval and answer evaluation with Weave
- `part3_1_eval` - Register the Advanced RAG dataset and evaluate `RagModel`
- `partX_1_langgraph_basics` - LangGraph basics (parked, not in the current course flow)

Notes on models: part1 / partX (pages, notebooks, `app/agent_loop.py`) use `gpt-5.6-luna`; part2 / part3 and `app/advanced_rag/` (answer, LLM rerank, judge, vision extractor) use `gpt-6-luna`, with `text-embedding-3-small` for embeddings. With the Chat Completions API, GPT-5.6 / GPT-6 accept function tools only with `reasoning_effort="none"`.

## Key Technical Details

- **Python 3.13**, managed with **uv** (dependencies in `pyproject.toml`, lock in `uv.lock`)
- **LangChain** + **LangGraph** for chains and agent orchestration
- **Streamlit** for web UI with `st.write_stream()` for streaming responses
- **LanceDB** persisted at `data/lancedb/`, with a `simple_rag` table for the introductory example and `docs_<extractor>` tables for Advanced RAG; OpenAI `text-embedding-3-small` is the default embedding model
- **Weave** (Weights & Biases) for tracing and evaluation
- Environment variables loaded from `.env` (see `.env.template`): `OPENAI_API_KEY`, `WANDB_API_KEY`, `WANDB_PROJECT`
- **Web search** via Amazon Bedrock AgentCore Gateway Web Search Tool (`app/tools/web_search.py`): exposes a plain `web_search()` function with no LangChain dependency; callers wrap it themselves (`@tool` in pages/notebooks or `function_to_tool` for the agent loop). Reads the gateway URL from `AGENTCORE_GATEWAY_URL` and the tool name from `AGENTCORE_WEB_SEARCH_TOOL_NAME` (both set by the hands-on environment; the SigV4 region is derived from the URL) and signs requests with AWS credentials (EC2 instance role); no API key needed
