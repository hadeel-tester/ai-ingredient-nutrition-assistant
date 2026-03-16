"""components.py — Reusable Streamlit UI components.

Contains small, self-contained widgets used across multiple pages.
"""

from __future__ import annotations

import streamlit as st


_TRAFFIC_LIGHT_BADGE: dict[str, str] = {
    "green": "🟢",
    "amber": "🟠",
    "red":   "🔴",
}

_NUTRIENT_UNITS: dict[str, str] = {
    "calories":      "Calories (kcal)",
    "fat":           "Fat (g)",
    "saturated_fat": "Saturated fat (g)",
    "sugar":         "Sugar (g)",
    "protein":       "Protein (g)",
    "salt":          "Salt (g)",
    "fibre":         "Fibre (g)",
}


def _render_nutrition_table(result: dict) -> None:
    """Render evaluate_nutrition output as a 4-column table (per 100g).

    Back-calculates per-100g values from per_serving ÷ serving_size_g × 100,
    then builds a markdown table with Nutrient | Per 100g | % Daily Intake | Rating.

    Args:
        result: Dict returned by evaluate_nutrition(), containing per_serving,
                dri_percent, traffic_lights, serving_size_g, overall, summary.
    """
    serving_size_g: float = result.get("serving_size_g", 0)
    if not serving_size_g:
        st.json(result)
        return

    per_serving: dict[str, float] = result.get("per_serving", {})
    dri_percent: dict[str, float] = result.get("dri_percent", {})
    traffic_lights: dict[str, str] = result.get("traffic_lights", {})

    rows = ["| Nutrient | Per 100g | % Daily Intake | Rating |",
            "|---|---|---|---|"]
    for key, label in _NUTRIENT_UNITS.items():
        value = per_serving.get(key, 0)
        per_100g = round(value / serving_size_g * 100, 1)
        dri = dri_percent.get(key, 0)
        colour = traffic_lights.get(key, "")
        badge = _TRAFFIC_LIGHT_BADGE.get(colour, "⚪")
        rows.append(f"| {label} | {per_100g} | {dri}% | {badge} |")

    st.markdown("\n".join(rows))
    reference = result.get("reference", "")
    if reference and not reference.startswith("EU DRI"):
        st.caption(f"📊 DRI% calculated using: {reference}")
    st.caption(
        f"**Overall: {result.get('overall', '—').capitalize()}** — "
        f"{result.get('summary', '')}"
    )


def _render_traffic_lights(traffic_lights: dict[str, str]) -> None:
    """Render a traffic_lights dict as coloured emoji badges.

    Args:
        traffic_lights: Mapping of nutrient name → 'green' | 'amber' | 'red'.
    """
    for nutrient, colour in traffic_lights.items():
        badge = _TRAFFIC_LIGHT_BADGE.get(colour, "⚪")
        label = nutrient.replace("_", " ").capitalize()
        st.markdown(f"{badge} **{label}** — {colour}")


def tool_result_card(tool_name: str, result: dict) -> None:
    """Render a styled card showing the output of a tool call.

    Traffic light values from the nutrition calculator are rendered as coloured
    emoji badges (🟢/🟠/🔴); all other tool results fall back to st.json.

    Args:
        tool_name: Display name for the tool (e.g. "🔍 Open Food Facts").
        result:    Dict returned by the tool.
    """
    with st.expander(f"Result — {tool_name}", expanded=True):
        traffic_lights: dict[str, str] | None = result.get("traffic_lights")
        if (traffic_lights and isinstance(traffic_lights, dict)
                and "per_serving" in result and "dri_percent" in result):
            # evaluate_nutrition output — render as table
            _render_nutrition_table(result)
        elif traffic_lights and isinstance(traffic_lights, dict):
            # Other tools with traffic lights — JSON + badge fallback
            rest = {k: v for k, v in result.items() if k != "traffic_lights"}
            if rest:
                st.json(rest)
            st.markdown("**Traffic lights**")
            _render_traffic_lights(traffic_lights)
        else:
            st.json(result)


def rag_process_expander(rag_chunks: list[dict]) -> None:
    """Render a collapsed expander showing all retrieved chunks with scores.

    Displays each chunk's ingredient name, section heading, similarity score,
    and whether it was used in the LLM context (✅) or filtered out (⬜).

    Collapsed by default so it doesn't dominate the chat UI — power users can
    expand it to inspect exactly what the RAG pipeline retrieved and why.

    Args:
        rag_chunks: List of dicts with keys:
            - ingredient (str): ingredient name from metadata
            - section    (str): section heading from metadata
            - score      (float): similarity score in [0, 1]
            - is_relevant (bool): True if the chunk passed _matches_query()
    """
    with st.expander("🔬 RAG process", expanded=False):
        if not rag_chunks:
            st.caption("No chunks retrieved from the knowledge base.")
            return

        for chunk in rag_chunks:
            ingredient = chunk.get("ingredient", "?").replace("_", " ").title()
            section = chunk.get("section", "?")
            score = chunk.get("score", 0.0)
            is_relevant = chunk.get("is_relevant", False)

            relevance_label = "✅ used" if is_relevant else "⬜ filtered out"
            score_pct = f"{score:.0%}"

            st.markdown(
                f"**{ingredient}** · `{section}` · {score_pct} match · {relevance_label}"
            )
