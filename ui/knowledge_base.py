"""knowledge_base.py — Knowledge base browser page.

Displays all ingredients currently stored in ChromaDB as a grid of cards.
Each card shows:
  - Ingredient display name
  - Risk level badge (colour-coded: red=high, orange=moderate, green=low)
  - EU regulatory status
  - Category

The data is read directly from ChromaDB metadata — no LLM call required.
"""

from __future__ import annotations

import streamlit as st
from langchain_chroma import Chroma

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Risk level → (emoji, hex colour) for the inline HTML badge.
_RISK_BADGE: dict[str, tuple[str, str]] = {
    "high":     ("🔴", "#d62728"),
    "moderate": ("🟠", "#e67e22"),
    "low":      ("🟢", "#27ae60"),
}
_RISK_BADGE_DEFAULT: tuple[str, str] = ("⚪", "#7f8c8d")

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _risk_badge_html(risk: str) -> str:
    """Return an inline HTML span styled as a colour-coded risk badge.

    Args:
        risk: Risk level string from metadata (e.g. "high", "moderate", "low").

    Returns:
        HTML string for st.markdown(..., unsafe_allow_html=True).
    """
    emoji, colour = _RISK_BADGE.get(risk.lower(), _RISK_BADGE_DEFAULT)
    label = risk.upper() if risk else "UNKNOWN"
    return (
        f'<span style="background:{colour};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:0.78em;font-weight:600">'
        f"{emoji} {label}</span>"
    )


def _ingredient_card(name: str, meta: dict) -> None:
    """Render a single ingredient card inside a bordered container.

    Args:
        name: Ingredient key from metadata (e.g. "artificial_flavours").
        meta: Full metadata dict for this ingredient.
    """
    display_name = name.replace("_", " ").title()
    risk = meta.get("risk_level", "") or ""
    category = meta.get("category", "") or "—"
    eu_status = meta.get("eu_status", "") or "—"
    aliases_raw = meta.get("aliases", "") or ""
    aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()]

    with st.container(border=True):
        st.markdown(f"#### {display_name}")
        st.markdown(_risk_badge_html(risk), unsafe_allow_html=True)
        st.caption(f"**Category:** {category}")
        st.caption(f"**EU Status:** {eu_status}")
        if aliases:
            # Show up to 4 aliases to keep cards compact
            alias_str = ", ".join(aliases[:4])
            if len(aliases) > 4:
                alias_str += f" +{len(aliases) - 4} more"
            st.caption(f"**Also known as:** {alias_str}")


def _load_unique_ingredients(vectorstore: Chroma) -> dict[str, dict]:
    """Query all ChromaDB metadata and deduplicate by ingredient name.

    Each ingredient may have multiple chunks (one per ## section). We keep
    only the first occurrence per ingredient name for the card grid.

    Args:
        vectorstore: Chroma instance to query.

    Returns:
        Dict mapping ingredient name → representative metadata dict,
        sorted alphabetically by ingredient name.
    """
    data = vectorstore.get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for meta in data.get("metadatas", []):
        name = meta.get("ingredient", "").strip()
        if name and name not in seen:
            seen[name] = meta
    return dict(sorted(seen.items()))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_kb_page(vectorstore: Chroma) -> None:
    """Render the knowledge base ingredient grid.

    Args:
        vectorstore: Chroma instance (passed from app.py cache).
    """
    st.header("📚 Knowledge Base")
    st.caption(
        "Ingredients currently stored in ChromaDB. "
        "Rebuild the knowledge base by running: `python -m knowledge_base.build_kb`"
    )

    ingredients = _load_unique_ingredients(vectorstore)

    if not ingredients:
        st.warning(
            "No ingredients found in ChromaDB. "
            "Run `python -m knowledge_base.build_kb` to populate the knowledge base."
        )
        return

    # Summary metrics row
    risk_counts: dict[str, int] = {"high": 0, "moderate": 0, "low": 0, "other": 0}
    for meta in ingredients.values():
        risk = (meta.get("risk_level") or "").lower()
        if risk in risk_counts:
            risk_counts[risk] += 1
        else:
            risk_counts["other"] += 1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total ingredients", len(ingredients))
    m2.metric("🔴 High risk", risk_counts["high"])
    m3.metric("🟠 Moderate risk", risk_counts["moderate"])
    m4.metric("🟢 Low risk", risk_counts["low"])

    st.divider()

    # Filter controls
    col_search, col_risk, col_cat = st.columns([2, 1, 1])
    with col_search:
        search_term = st.text_input("🔍 Search ingredients", placeholder="e.g. aspartame")
    with col_risk:
        risk_filter = st.selectbox(
            "Risk level", options=["All", "High", "Moderate", "Low"]
        )
    with col_cat:
        all_categories = sorted(
            {
                (meta.get("category") or "Unknown").strip().title()
                for meta in ingredients.values()
            }
        )
        cat_filter = st.selectbox("Category", options=["All"] + all_categories)

    # Apply filters
    filtered = {
        name: meta
        for name, meta in ingredients.items()
        if (
            (not search_term or search_term.lower() in name.lower()
             or search_term.lower() in (meta.get("aliases") or "").lower())
            and (risk_filter == "All"
                 or (meta.get("risk_level") or "").lower() == risk_filter.lower())
            and (cat_filter == "All"
                 or (meta.get("category") or "").strip().title() == cat_filter)
        )
    }

    if not filtered:
        st.info("No ingredients match your filters.")
        return

    st.caption(f"Showing {len(filtered)} of {len(ingredients)} ingredients")
    st.divider()

    # 3-column card grid
    cols = st.columns(3)
    for i, (name, meta) in enumerate(filtered.items()):
        with cols[i % 3]:
            _ingredient_card(name, meta)
