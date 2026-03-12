"""ingredient_analyzer.py — LangChain tool for KB-backed ingredient health analysis.

Given a raw ingredients string (e.g. from a product label or Open Food Facts),
parses individual ingredients and queries the ChromaDB knowledge base for each.
Returns structured health information for every ingredient that has a KB entry.

This tool bridges the gap between live product data (Open Food Facts) and the
curated knowledge base — it lets the agent automatically enrich product lookups
with health context per ingredient.
"""

from __future__ import annotations

import re

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from rag.vectorstore import get_vectorstore

_NA = "not available"


class AnalyzeIngredientsInput(BaseModel):
    """Input schema for the ingredient analysis tool."""

    ingredients_text: str = Field(
        description=(
            "Raw ingredients string from a product label or Open Food Facts lookup. "
            "Comma-separated list, e.g. 'sugar, palm oil, cocoa, aspartame, lecithin'."
        )
    )


def _parse_ingredients(text: str) -> list[str]:
    """Split a raw ingredients string into cleaned individual ingredient names.

    Handles comma-separated lists, strips percentages, parenthetical sub-ingredients,
    and normalises whitespace. Deduplicates while preserving order.

    Args:
        text: Raw ingredients text from a product label.

    Returns:
        Deduplicated list of lowercase ingredient name strings.
    """
    # Remove percentages like "sugar 10%"
    text = re.sub(r"\d+(\.\d+)?\s*%", "", text)
    # Remove parenthetical content like "lecithin (soy)"
    text = re.sub(r"\([^)]*\)", "", text)
    # Split on commas, semicolons, or period-separated lists
    parts = re.split(r"[,;.]", text)

    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        name = part.strip().lower()
        # Skip empty strings and very short fragments
        if len(name) < 2:
            continue
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _query_kb_for_ingredient(
    ingredient: str, vectorstore, k: int = 2
) -> list[dict] | None:
    """Query the knowledge base for a single ingredient.

    Uses similarity search with a low k to find the most relevant chunks.
    Validates that the returned chunks actually match the queried ingredient
    by checking the metadata (ingredient name and aliases).

    Args:
        ingredient: Lowercase ingredient name to search for.
        vectorstore: ChromaDB Chroma instance.
        k: Number of chunks to retrieve.

    Returns:
        List of matching chunk dicts with keys: ingredient, section, risk_level,
        content. Returns None if no relevant chunks are found.
    """
    docs_with_scores = vectorstore.similarity_search_with_score(ingredient, k=k)

    matches: list[dict] = []
    for doc, distance in docs_with_scores:
        metadata = doc.metadata
        kb_ingredient = metadata.get("ingredient", "").lower()
        aliases_raw = metadata.get("aliases", "")
        aliases = [a.strip().lower() for a in aliases_raw.split(",") if a.strip()]

        # Check if the queried ingredient matches the KB entry
        candidates = [kb_ingredient] + aliases
        is_match = any(
            (c in ingredient or ingredient in c)
            for c in candidates
            if c
        )

        if is_match:
            matches.append({
                "ingredient": metadata.get("ingredient", _NA),
                "section": metadata.get("section", _NA),
                "risk_level": metadata.get("risk_level", _NA),
                "category": metadata.get("category", _NA),
                "content": doc.page_content[:500],
            })

    return matches if matches else None


@tool(args_schema=AnalyzeIngredientsInput)
def analyze_product_ingredients(ingredients_text: str) -> dict:
    """Analyze ingredients from a product against the curated knowledge base.

    Parses the ingredients list and looks up each ingredient in the knowledge base.
    Returns health information, risk levels, and key findings for ingredients that
    have entries in the KB. Ingredients not found in the KB are listed separately.

    Use this tool after looking up a product to provide health context for its ingredients.

    Args:
        ingredients_text: Raw ingredients string, comma-separated.

    Returns:
        Dict with keys:
        - found: list of dicts with ingredient health info from the KB
        - not_in_kb: list of ingredient names not found in the knowledge base
        - summary: short summary string
    """
    ingredients = _parse_ingredients(ingredients_text)
    if not ingredients:
        return {"error": "Could not parse any ingredients from the provided text."}

    vectorstore = get_vectorstore()

    found: list[dict] = []
    not_in_kb: list[str] = []
    seen_kb_ingredients: set[str] = set()

    for ingredient in ingredients:
        matches = _query_kb_for_ingredient(ingredient, vectorstore)
        if matches:
            for match in matches:
                kb_name = match["ingredient"].lower()
                if kb_name not in seen_kb_ingredients:
                    seen_kb_ingredients.add(kb_name)
                    found.append(match)
        else:
            not_in_kb.append(ingredient)

    summary_parts: list[str] = []
    if found:
        kb_names = list({m["ingredient"] for m in found})
        summary_parts.append(
            f"{len(kb_names)} ingredient(s) found in knowledge base: "
            f"{', '.join(kb_names)}"
        )
        high_risk = [m["ingredient"] for m in found if m.get("risk_level") == "high"]
        if high_risk:
            summary_parts.append(
                f"⚠️ High-risk ingredients: {', '.join(high_risk)}"
            )
    else:
        summary_parts.append("No ingredients found in the knowledge base.")

    if not_in_kb:
        summary_parts.append(
            f"{len(not_in_kb)} ingredient(s) not in KB (no health data available)."
        )

    return {
        "found": found,
        "not_in_kb": not_in_kb,
        "summary": " | ".join(summary_parts),
    }
