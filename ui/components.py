"""components.py — Reusable Streamlit UI components.

Contains small, self-contained widgets used across multiple pages.
"""

from __future__ import annotations

import streamlit as st


def tool_result_card(tool_name: str, result: dict) -> None:
    """Render a styled card showing the output of a tool call.

    Args:
        tool_name: Display name for the tool (e.g. "🔍 Open Food Facts").
        result:    Dict returned by the tool.
    """
    with st.expander(f"Result — {tool_name}", expanded=True):
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
