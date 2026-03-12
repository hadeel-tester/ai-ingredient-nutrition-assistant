# CLAUDE.md — Project Context for Claude Code

This file is read by Claude Code at the start of every session.
It provides persistent context so Claude Code understands who I am,
how I work, and how this project should be structured.

---

## About This Developer

- Background: QA engineering — strong instincts for testing, edge cases, and validation
- Currently: AI Engineering student (Turing College)
- Goal: Build portfolio-ready, production-quality projects — not just demos
- Tools: Cursor + Claude Code for development
- Learning style: Explain WHY patterns are used, not just what they do

---

## Current Sprint / Project

**Sprint:** Sprint 2 — Domain-Focused RAG Chatbot  
**Requirements file:** `project_requirements.md` (read this for full task details)

**Project:** Ingredient & Nutrition Chatbot  
**Domain:** Food safety and nutrition information

**Core deliverables:**
- Advanced RAG with query translation and structured retrieval
- At least 3 tool calls (domain-relevant)
- Streamlit UI with source citations and tool call results displayed

**Stack additions for this sprint:**
- LangChain
- Vector database (e.g., ChromaDB or FAISS)
- OpenAI embeddings (or Anthropic-compatible alternative)

**Optional tasks being targeted:**
- [ ] Source citations in responses (easy)
- [ ] Connect to a remote MCP server (medium)
- [ ] Source citations in responses (medium)
- [ ] Implement tools as MCP servers (hard)

---


## Project Stack

> ⚠️ Update this section per project. Only list what is actually installed and used.

**Always present:**
- Python 3.10+
- Anthropic Claude API (official `anthropic` SDK)
- `python-dotenv` for environment variables- `pytest` for testing


---

## How to Work With Me

- I am learning — explain WHY a pattern is used, not just what it does
- When multiple approaches exist, briefly list tradeoffs before picking one
- Point out edge cases and validation gaps — lean into the QA background
- Don't skip error handling to keep examples short — write the real implementation
- Don't suggest no-code solutions — hands-on coding is the priority

---

## Architecture Philosophy

This is the most important section. Every project I build is designed
with future reuse in mind — not as a one-off solution.

- **Separate core logic from domain-specific implementation**
  The engine (what it does) should be decoupled from the skin (what industry it's for)
- **Avoid hardcoding domain-specific values** — use config files, constants, or parameters
- **Functions and modules should be easy to lift out** and drop into a new project
  with minimal changes
- **When building a feature, ask:** could this work in a different industry or context?
  If yes, structure it that way from the start
- **Three-layer thinking:**
  1. `/core` — reusable engine (no domain knowledge)
  2. `/features` — domain-specific logic that uses the engine
  3. `/app` or `/ui` — the presentable interface

---

## Python Standards

- PEP 8 style
- Type hints on all function signatures
- Docstrings on all functions and classes
- Prefer `dataclasses` or `Pydantic` models over raw dicts for structured data
- Use `pathlib` over `os.path`
- Handle exceptions explicitly — never bare `except:`
- Always assume a virtual environment is active

---

## AI / LLM Standards

- Define system prompts as separate string constants, not inline
- Structure prompts clearly: system role → context → task → output format
- Add retry logic for API calls
- Never hardcode API keys — use `.env` + `python-dotenv`
- Store all prompt templates in a `/prompts` folder
- Log token usage where possible for cost awareness
- For RAG/retrieval: chunk thoughtfully, not just by character count

---

## Testing Standards

- Use `pytest`
- Write tests alongside features, not after
- Cover: happy path + at least 2 edge cases per function
- For AI outputs: assert non-empty response and expected structure
- Mock all external API calls in the test suite

---

## Folder Structure

```
my-project/
├── .env                  ← never commit this
├── CLAUDE.md             ← this file
├── .cursorrules          ← Cursor context
│
├── core/                 ← reusable engines, no domain knowledge
├── features/             ← domain-specific logic
│   └── example_feature/
│       ├── service.py
│       └── types.py
├── prompts/              ← all LLM prompt templates
├── utils/                ← small shared helpers
└── tests/                ← mirrors the structure above
```

---

## What to Avoid

- No placeholder TODO code — write the real implementation
- No `any` type in TypeScript unless absolutely necessary
- No deprecated libraries or APIs
- No bare `except:` in Python
- No hardcoded secrets, API keys, or domain-specific magic values
