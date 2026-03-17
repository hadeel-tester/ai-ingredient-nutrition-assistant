# NutriMind AI — Ingredient & Nutrition Assistant

An AI-powered food product analysis chatbot built with LangChain, RAG, and Streamlit.
Ask about any food product by name or barcode and get a full breakdown: ingredient health risks, EU allergens, nutritional evaluation, and an overall health score — all personalised to your dietary profile.

---

## Why This Project

Most food label apps stop at displaying raw data — calories, macros, a Nutri-Score. NutriMind AI goes further: it explains *what the ingredients actually mean for your health*.

Built as part of an AI Engineering programme, this project combines a live product database (Open Food Facts) with a curated ingredient knowledge base to provide health context, EU regulatory status, and risk assessments that generic nutrition apps don't offer. The architecture is designed for reuse — the ingredient analysis engine, allergen checker, and nutrition tools are modular components that will power a future mobile QR scanner app for real-time product health analysis at the point of purchase.

---

## Features

- **Live product lookup** via Open Food Facts API (barcode or name)
- **Ingredient health analysis** backed by a curated knowledge base (38 ingredients)
- **EU allergen detection** — all 14 regulated allergens with E-number recognition
- **Nutritional evaluation** with UK FSA traffic light ratings and % daily intake
- **Health scoring** — 0–100 score with A–F grade, positives/negatives breakdown
- **User profile personalisation** — allergens, dietary preferences, health goals, calorie target
- **RAG pipeline** — ChromaDB vector store with query relevance filtering
- **Source citations** — every response shows which KB documents were used
- **Session cost tracking** — live token usage and estimated cost displayed in sidebar
- **Rate limiting** — 3-second cooldown between requests and a 50-request session cap to protect API quotas

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | GPT-4o-mini (OpenAI) |
| Orchestration | LangChain |
| Vector store | ChromaDB |
| Embeddings | OpenAI `text-embedding-3-small` |
| Retry logic | Tenacity |
| Validation | Pydantic v2 |
| Testing | pytest |

---

## Project Structure

```
ai-ingredient-nutrition-assistant/
├── app.py                      # Streamlit entry point
├── requirements.txt
│
├── chains/
│   └── chat_chain.py           # LangChain agent + mandatory tool chaining
│
├── knowledge_base/
│   ├── build_kb.py             # Builds ChromaDB from markdown documents
│   └── documents/              # Markdown ingredient files (source of truth)
│
├── rag/
│   ├── retriever.py            # Query translation + relevance filtering
│   └── vectorstore.py          # ChromaDB connection factory
│
├── tools/
│   ├── open_food_facts.py      # Live product lookup (Open Food Facts API)
│   ├── ingredient_analyzer.py  # KB-backed per-ingredient health analysis
│   ├── allergen_checker.py     # EU allergen keyword matching
│   ├── nutrition_calculator.py # Per-serving DRI evaluation + traffic lights
│   └── health_scorer.py        # 0–100 health score with A–F grade
│
├── ui/
│   ├── chat.py                 # Chat page
│   ├── components.py           # Reusable widgets (nutrition table, tool cards)
│   ├── profile.py              # User profile page
│   └── knowledge_base.py       # KB browser page
│
├── utils/
│   └── token_tracker.py        # Session-level token usage and cost tracker
│
└── tests/                      # pytest suite mirroring the source structure
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd ai-ingredient-nutrition-assistant
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

### 4. Build the knowledge base

This only needs to run once, or whenever you add/update documents in `knowledge_base/documents/`.

```bash
python -m knowledge_base.build_kb
```

This loads all `.md` files, parses YAML frontmatter, chunks by section, embeds with OpenAI, and persists to ChromaDB.

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Usage

### Chat page

Type a question or use the **Quick product lookup** sidebar to search by name or barcode.

Example queries:
- `Look up Nutella`
- `3017620422003` (Nutella barcode)
- `Is aspartame safe?`
- `Check allergens in: wheat flour, sugar, whey, palm oil, soy lecithin`
- `What are the health risks of carrageenan?`

When a product is found, the assistant automatically runs all 5 analysis tools in sequence and presents a structured response:

1. **Product Data** — name, Nutri-Score, NOVA group
2. **Ingredient Health Analysis** — risk levels from the knowledge base
3. **Allergen Report** — confirmed and possible EU allergens
4. **Nutritional Evaluation** — interactive table (per 100g, % DRI, traffic lights)
5. **Health Score** — 0–100 score with grade and recommendation
6. **Conclusion** — personalised verdict based on your profile

### Profile page

Set your allergens, dietary preferences, health goals, and daily calorie target. The assistant factors these into every response and warns when a product conflicts with your profile.

### Session cost tracking

The sidebar displays a live **💰 Session Cost** panel showing total tokens used and estimated cost (gpt-4o-mini pricing: $0.15/1M input, $0.60/1M output). Click **Reset cost tracker** to zero the counters.

---

## Running Tests

```bash
pytest
```

Or for a specific module:

```bash
pytest tests/tools/test_nutrition_calculator.py -v
```

---

## Adding Knowledge Base Documents

Add a new `.md` file to `knowledge_base/documents/` with the following frontmatter:

```markdown
---
ingredient: palm oil
category: fat
aliases: [palm fat, elaeis guineensis]
risk_level: moderate
e_number: null
eu_status: permitted
allergen: false
vegan: true
---

## Overview
...

## Health Risks
...
```

Then rebuild the knowledge base:

```bash
python -m knowledge_base.build_kb
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM and embeddings |
| `CHROMA_PERSIST_DIR` | No | ChromaDB path (default: `./chroma_db`) |

---

## Rate Limiting

The app enforces two session-level guards (configurable via constants in `ui/chat.py`):

| Limit | Default | Purpose |
|---|---|---|
| Cooldown | 3 seconds | Prevents rapid-fire double submissions |
| Session cap | 50 requests | Protects OpenAI quota during demos |

When either limit is hit, the UI shows an inline warning instead of dispatching the chain. Refresh the page to reset both counters. For production deployments, replace session state with a shared backend store (Redis, etc.).

---

## Known Limitations

- **Knowledge base coverage** — The curated ingredient KB currently covers 30+ common additives and oils. Ingredients outside this set (e.g. specific flavourings, enzymes, or novel additives) return a general knowledge response clearly marked as unverified. Coverage grows with each sprint.
- **Ingredient parsing** — The ingredient parser handles most EU-format labels but may miss deeply nested sub-ingredients or non-standard formatting from some Open Food Facts entries.
- **Serving size defaults** — Nutritional evaluation currently uses per-100g as the reference unit. Product-specific serving sizes (e.g. 330ml for a can, 30g for cereal) are not yet dynamically inferred from product data.
- **Saturated fat data gaps** — Some Open Food Facts entries don't include `saturated-fat_100g` in their nutriment data, causing the field to show 0g even when the product contains saturated fat.
- **Single-user architecture** — Rate limiting uses Streamlit session state, which works for demos and single-user deployments. A production multi-user deployment would require a shared backend store (Redis or similar) for rate limiting and user profiles.
- **LLM non-determinism** — Tool chaining is enforced via the system prompt. Occasionally the LLM may not call all tools for ambiguous queries, though mandatory chaining instructions mitigate this for product lookups.

---

## Reflection and Improvement

### What worked well

The three-layer separation — live API data, curated KB context, and LLM reasoning — proved to be the right architecture. Open Food Facts provides raw product data; the RAG knowledge base provides health meaning; the LLM synthesises both into a personalised response. Each layer can be improved independently without touching the others.

Enforcing deterministic tool chaining via the system prompt rather than hardcoded orchestration kept the agent genuinely agentic for freeform queries while ensuring consistent output for product lookups.

### What I would do differently

The ingredient parser's initial assumption — that parenthetical content is always a source clarification (e.g. `"lecithin (soy)"`) — turned out to be wrong for many real-world labels where the actual ingredients live inside brackets (e.g. `"sweeteners (aspartame, acesulfame K)"`). A more robust parser that expands parenthetical sub-ingredients from the start would have avoided several downstream issues with the ingredient analyzer and health scorer.

### Planned improvements

- Wrap the analysis engine in a FastAPI backend to support mobile app integration
- Add CIQUAL micronutrient data (vitamins, minerals) to the knowledge base for Sprint 3's meal planning agent
- Implement barcode scanning via device camera for the mobile QR Scanner app
- Expand KB coverage to 100+ ingredients with automated freshness checks against EFSA updates
- Replace session-state rate limiting with Redis for multi-user production deployments

---

## License

MIT
