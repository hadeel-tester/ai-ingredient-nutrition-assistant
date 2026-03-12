"""sources.py — Knowledge base viewer page.

Lets users browse and inspect the documents loaded into ChromaDB.
"""

from __future__ import annotations

import streamlit as st


def render_sources_page() -> None:
    """Render the knowledge base document viewer."""
    st.title("Knowledge Base")
    st.caption("Documents and data sources loaded into the vector store.")

    # TODO: query ChromaDB collection metadata and display source documents
    st.info("Knowledge base viewer coming soon.")
