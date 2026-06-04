# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Source code for the AI Agent Developer Training Course (AIエージェント開発者養成講座). This is teaching material / hands-on courseware, not production software. All documentation and comments are in Japanese.

## Repository Layout

The repo is organized by course day, with each day having a complete ("instructor") version and a generated "starter" version handed to students:

| Directory | Role |
|-----------|------|
| `day2/` | Python / Streamlit material: RAG, agents, LangGraph, Jupyter notebooks. See `day2/CLAUDE.md`. |
| `day2-starter/` | **Generated** student starter for day2. Do not edit directly. |
| `day3/` | Next.js + FastAPI customer-inquiry app (お問い合わせ対応). See `day3/CLAUDE.md`. |
| `day3-starter/` | **Generated** student starter for day3. Do not edit directly. |
| `setup/` | Hands-on environment setup (AWS EC2 + code-server). See `setup/README.md`. |
| `docs/` | Course prep docs and API-key acquisition guides (Cohere, Tavily, Weave, Azure OpenAI). |
| `scripts/` | Starter-generation scripts (see below). |

**Each day directory has its own `CLAUDE.md` with detailed architecture and commands — read it when working inside that directory.** This root file only covers cross-cutting structure.

## Starter Generation (critical)

`day2-starter/` and `day3-starter/` are **fully regenerated** from `day2/` and `day3/` — never edit a starter by hand; the next `make build` will wipe it (`rm -rf` of the whole starter dir).

```bash
make build   # regenerates both starters from day2/ and day3/
```

To change what students receive:

- **Edit the source** (`day2/` or `day3/`), then run `make build`.
- **Control which files are included** via `scripts/generate-day{2,3}-starter/include.txt`. Only paths listed there are copied into the starter; anything not listed is excluded. This is how exercise files are withheld from students (e.g. day3 includes only `evals/__init__.py` so students implement the rest).
- **Override a file for the starter only** by placing it under `scripts/generate-day{2,3}-starter/overrides/` mirroring the target path. Overrides are applied last (after the include copy), and typically blank out solution code so students fill it in (e.g. `day3-starter`'s `generate_response.py`).

After editing any source file, run `make build` and verify the starter diff before committing.
