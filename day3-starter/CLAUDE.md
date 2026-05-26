# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-assisted customer inquiry management system (お問い合わせ対応) built with Next.js 16 + React 19. Customers submit inquiries via a public form; an LLM service (FastAPI+LangGraph) generates draft responses asynchronously; staff review, edit, and send responses via an admin dashboard. All UI text is in Japanese.

## Commands

### Next.js (web/)

```bash
cd web
npm run dev      # Start dev server at localhost:3000
npm run build    # Production build
npm start        # Run production server
npm run lint     # ESLint (Next.js + TypeScript rules)
npm run seed     # Seed database with sample data (clears existing data first)
```

No test framework is currently configured.

### FastAPI (llm-app/)

```bash
cd llm-app
uv sync                                          # Install dependencies
uv run uvicorn app.main:app --reload --port 8000  # Start dev server at localhost:8000
uv run ruff check .                              # Lint Python code
uv run mypy .                                    # Type check
```

Requires `ANTHROPIC_API_KEY` in `llm-app/.env` (see `.env.example`). Optional: `ANTHROPIC_MODEL` (defaults to claude-sonnet-4-5-20250929).

The Next.js app connects to FastAPI via `LLM_API_URL` env var (defaults to `http://localhost:8000`).

## Architecture

### Two services

- **web/** — Next.js 16 App Router: public contact form (top page `/`), admin dashboard (`/admin`), 7 API routes under `src/app/api/`, SQLite database via better-sqlite3 (WAL mode, auto-creates at `web/data/inquiries.db`)
- **llm-app/** — FastAPI + LangGraph: `POST /api/generate` runs the AI workflow

### Key files

- `web/src/lib/db.ts` — Database singleton, schema initialization, all CRUD operations; defines TypeScript types (`Inquiry`, `InquiryStatus`, `InquiryTopic`, `QualityScores`, `AIResponse`)
- `web/src/lib/llm.ts` — Calls FastAPI backend with 60s timeout; quality alert = politeness NG
- `web/src/app/api/inquiries/route.ts` — Uses Next.js 16 `after()` to trigger AI generation in background after returning immediate 200 to customer
- `llm-app/app/generate/graph.py` — LangGraph state machine definition and `GraphState` TypedDict

### LangGraph workflow (llm-app/)

3-node pipeline with conditional routing (`llm-app/app/generate/nodes/`):

```
classify_topic → [spam?] → END
                 [else] → generate_response → quality_check → END
```

- `classify_topic.py` — Claude classifies inquiry into product/development/other/spam with confidence score
- `generate_response.py` — Claude generates draft response (skipped for spam)
- `quality_check.py` — Claude evaluates politeness (OK/NG; skipped for spam)

### Inquiry lifecycle

`processing` → `draft` → `sent`

1. Customer submits form → inquiry saved as `processing`, immediate 200 response
2. `after()` triggers LLM → AI generates response with quality scores → status becomes `draft`
3. Staff reviews on admin dashboard (auto-refreshes every 5s) → edits if needed → sends → `sent`

### API routes

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/inquiries` | Submit new inquiry (public) |
| GET | `/api/admin/inquiries` | List inquiries with status/topic filters, pagination |
| GET | `/api/admin/inquiries/[id]` | Get inquiry detail |
| POST | `/api/admin/inquiries/[id]/draft` | Save edited draft |
| POST | `/api/admin/inquiries/[id]/send` | Send response |
| POST | `/api/admin/inquiries/[id]/topic` | Update topic classification |

### UI stack

shadcn/ui components (in `web/src/components/ui/`) built on Radix UI primitives, styled with Tailwind CSS v4.

## Path Aliases

TypeScript path alias `@/*` maps to `web/src/*` (configured in tsconfig.json).

## Project Notes

- このリポジトリは研修・ハンズオン教材であり、本番運用ではない。DB (`web/data/inquiries.db`) は消して再シードする選択を躊躇なく取ってよい。後方互換シム、ALTER TABLE による既存DBマイグレーション、廃止フィールドのエイリアスなどは書かない。スキーマ変更は `web/src/lib/db.ts` の `initSchema` を直接書き換えて再シードで対応する。
- 永続化したい知見・ルールはこの `CLAUDE.md` に追記する方針(memory システムは使わない)。
