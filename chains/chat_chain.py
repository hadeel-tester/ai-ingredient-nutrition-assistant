"""chat_chain.py — LangChain agent connecting RAG retrieval with tools.

Builds a conversational tool-calling agent that:
1. Pre-fetches relevant nutrition context from ChromaDB via the retriever
2. Injects that context into the system prompt
3. Lets the LLM autonomously invoke tools when needed:
   - lookup_product              — live Open Food Facts API lookup by barcode or name
   - analyze_product_ingredients — KB-backed health analysis per ingredient (auto-chained after lookup)
   - check_allergens             — EU allergen detection from an ingredient label
   - evaluate_nutrition          — per-serving DRI evaluation with traffic light ratings
   - score_health                — 0–100 health score with A–F grade

Usage:
    chain = build_chat_chain()
    result = chain.invoke({"input": "...", "chat_history": []})
    print(result["output"])
"""

from __future__ import annotations

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from rag.retriever import NO_RELEVANT_CONTEXT, get_retriever, retrieve_context
from tools.allergen_checker import check_allergens
from tools.health_scorer import score_health
from tools.ingredient_analyzer import analyze_product_ingredients
from tools.nutrition_calculator import evaluate_nutrition
from tools.open_food_facts import lookup_product

MODEL_NAME: str = "gpt-4o-mini"

# All tools available to the agent. The LLM decides which to call based on the
# question — no explicit routing logic needed.
TOOLS = [lookup_product, analyze_product_ingredients, check_allergens, evaluate_nutrition, score_health]

# Disclaimer appended programmatically when no KB context was found.
# Not left to the LLM — guaranteed to appear on every general-knowledge response.
_DISCLAIMER_TEXT: str = (
    "\n\n⚠️ This information is from general knowledge — it has not been verified "
    "against our curated knowledge base. For verified information, this ingredient "
    "will be added in a future update."
)

# ---------------------------------------------------------------------------
# Agent prompt
# ---------------------------------------------------------------------------
# create_tool_calling_agent requires four elements in this exact order:
#   1. system message          — sets persona + injects RAG context
#   2. chat_history            — conversation memory placeholder
#   3. human message           — current user input (key must be "input")
#   4. agent_scratchpad        — required placeholder for tool-call tracking

_AGENT_SYSTEM_PROMPT: str = """\
You are a nutrition and ingredient health assistant with access to tools and a curated knowledge base.

Use your tools when the user asks about:
  - A specific food product (look it up by name or barcode)
  - Allergens in an ingredient list
  - Per-serving nutritional values or daily reference intake percentages
  - A health score or grade for a product

IMPORTANT — mandatory tool chaining for product lookups:
Whenever lookup_product returns successfully (no "error" key), you MUST call ALL of the \
following tools in this exact order before generating your response. These are mandatory \
follow-up calls, not optional:

1. analyze_product_ingredients — pass the "ingredients" string from the lookup result.
2. check_allergens — pass the same "ingredients" string from the lookup result.
3. evaluate_nutrition — pass the nutriment values from the lookup result \
   (calories_per_100g, fat_per_100g, sugar_per_100g, protein_per_100g, salt_per_100g). \
   Set saturated_fat_per_100g and fibre_per_100g to 0 if not available. \
   If the user profile contains a daily calorie target, pass it as calorie_target.
4. score_health — pass the "ingredients" string and the same nutriment values \
   (calories_per_100g, fat_per_100g, sugar_per_100g, protein_per_100g, salt_per_100g). \
   Set saturated_fat_per_100g and fibre_per_100g to 0 if not available.

Never skip any of these steps. Never generate a product response without completing all four.

When presenting product information, clearly separate:
1. **Product Data** (from Open Food Facts): product name, nutritional values, Nutri-Score, \
NOVA group, allergens.
2. **Ingredient Health Analysis** (from knowledge base): risk levels, health findings, and \
safety notes for ingredients found in the KB. Mention which ingredients were NOT found in \
the KB.
3. **Allergen Report** (from allergen checker): confirmed and possible allergens detected.
4. **Nutritional Evaluation** (from nutrition calculator): write the section heading, \
then on its own line write exactly the text NUTRITION_TABLE_HERE (nothing else on that line), \
then one sentence summarising the key finding (e.g. "High in sugar and fat."). \
The UI will replace NUTRITION_TABLE_HERE with an interactive table automatically.
5. **Health Score** (from health scorer): overall score, grade, positives, negatives, and \
recommendation.
6. **Conclusion**: always end with this heading and 1–2 sentences summarising the overall \
verdict and any personalised recommendation based on the user's profile.

When answering questions, follow these sourcing rules:

1. If the answer is found in the knowledge base context provided below, answer from it \
and cite the sources mentioned in the context.

2. If the context is partial or only loosely related, combine it with your general \
knowledge and answer directly.

Always be accurate and cite uncertainty where it exists. Do not hallucinate nutritional values \
— use the tools and knowledge base instead.

Knowledge base context:
{context}

User profile:
{user_profile}

Factor the user's allergens, dietary preferences, and health goals into every response. \
Proactively warn if a product or ingredient conflicts with their profile (e.g. contains an \
allergen they listed, contradicts a dietary preference, or works against a health goal). \
If the profile says "No user profile set.", respond as usual without personalisation.
"""

AGENT_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", _AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_profile(profile: dict) -> str:
    """Format a user profile dict as a human-readable string for the system prompt.

    Args:
        profile: Dict with optional keys: name, allergens, preferences,
                 goals, calorie_target. Pass {} for an anonymous user.

    Returns:
        Multi-line string summarising the profile, or a "No profile set"
        sentinel if the dict is empty.
    """
    if not profile:
        return "No user profile set."
    parts: list[str] = []
    if name := profile.get("name"):
        parts.append(f"User name: {name}")
    if allergens := profile.get("allergens"):
        parts.append(f"Allergens to avoid: {', '.join(allergens)}")
    if preferences := profile.get("preferences"):
        parts.append(f"Dietary preferences: {', '.join(preferences)}")
    if goals := profile.get("goals"):
        parts.append(f"Health goals: {', '.join(goals)}")
    if target := profile.get("calorie_target"):
        parts.append(f"Daily calorie target: {target} kcal")
    return "\n".join(parts) if parts else "No user profile set."


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------


def build_chat_chain():
    """Build and return the RAG-augmented tool-calling agent chain.

    Chain flow:
        {"input": str, "chat_history": list, "user_profile": dict}
          → retrieve_context()  — fetches KB chunks; returns NO_RELEVANT_CONTEXT if none match
          → AgentExecutor loop  — LLM + tools
          → post-processing     — appends disclaimer if context was NO_RELEVANT_CONTEXT

    The disclaimer is appended in Python rather than relying on the LLM to include
    it, because LLMs do not reliably follow "always add this text" instructions.

    The returned chain accepts a dict with keys:
      - "input":        the user's message (str)
      - "chat_history": prior conversation turns (list of BaseMessage)
      - "user_profile": user profile dict (pass {} if not set)

    It returns a dict; extract the response with result["output"].

    Returns:
        A LangChain Runnable (RunnableLambda wrapping AgentExecutor).
    """
    retriever = get_retriever()
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2, streaming=True, stream_usage=True)

    agent = create_tool_calling_agent(llm, TOOLS, AGENT_PROMPT)
    agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=False,
                                   return_intermediate_steps=True)

    def _run(inputs: dict) -> dict:
        context = retrieve_context(inputs["input"], retriever)
        result = agent_executor.invoke({
            **inputs,
            "context": context,
            "user_profile": _format_profile(inputs.get("user_profile", {})),
        })
        if context == NO_RELEVANT_CONTEXT:
            result = dict(result)
            result["output"] = result["output"] + _DISCLAIMER_TEXT
        return result

    return RunnableLambda(_run)
