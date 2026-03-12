"""Unit tests for rag/retriever.py — _matches_query() and retrieve_context().

All tests are purely in-memory. The retriever is mocked with MagicMock so
no ChromaDB connection or OpenAI API call is needed.

Run with:
    pytest tests/rag/test_retriever.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rag.retriever import NO_RELEVANT_CONTEXT, _matches_query, retrieve_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(page_content: str = "some content", **metadata) -> Document:
    """Create a LangChain Document with the given metadata."""
    return Document(page_content=page_content, metadata=metadata)


def _make_retriever(docs: list[Document]) -> MagicMock:
    """Return a mock retriever whose .invoke() always returns docs."""
    retriever = MagicMock()
    retriever.invoke.return_value = docs
    return retriever


# ---------------------------------------------------------------------------
# _matches_query tests
# ---------------------------------------------------------------------------


def test_matches_query_ingredient_name():
    """Ingredient name found as substring in query → True."""
    meta = {"ingredient": "aspartame", "aliases": ""}
    assert _matches_query(meta, "is aspartame safe?") is True


def test_matches_query_alias_e_number():
    """E-number alias found in query → True."""
    meta = {"ingredient": "bha_bht", "aliases": "E320, BHA, Butylated hydroxyanisole"}
    assert _matches_query(meta, "what is e320?") is True


def test_matches_query_no_match():
    """Neither ingredient nor alias appears in query → False."""
    meta = {"ingredient": "bha_bht", "aliases": "E320, BHA"}
    assert _matches_query(meta, "tell me about palm oil") is False


def test_matches_query_underscore_normalisation():
    """Ingredient stored as 'artificial_flavours' matches 'artificial flavours' in query."""
    meta = {"ingredient": "artificial_flavours", "aliases": ""}
    assert _matches_query(meta, "artificial flavours risks") is True


def test_matches_query_case_insensitive():
    """Matching is case-insensitive on both sides."""
    meta = {"ingredient": "Aspartame", "aliases": "E951"}
    assert _matches_query(meta, "ASPARTAME dangers") is True


def test_matches_query_missing_metadata_fields():
    """Missing 'ingredient' and 'aliases' keys must not raise; returns False."""
    assert _matches_query({}, "aspartame") is False


# ---------------------------------------------------------------------------
# retrieve_context tests
# ---------------------------------------------------------------------------


def test_retrieve_context_returns_content_on_match():
    """When a chunk's ingredient matches the query, its content is returned."""
    doc = _make_doc("Aspartame summary text.", ingredient="aspartame", aliases="E951")
    retriever = _make_retriever([doc])

    result = retrieve_context("Is aspartame safe?", retriever)

    assert result == "Aspartame summary text."
    assert result != NO_RELEVANT_CONTEXT


def test_retrieve_context_sentinel_on_no_match():
    """When no chunk matches, NO_RELEVANT_CONTEXT is returned."""
    doc = _make_doc("BHA content.", ingredient="bha_bht", aliases="E320, BHA")
    retriever = _make_retriever([doc])

    result = retrieve_context("tell me about palm oil", retriever)

    assert result == NO_RELEVANT_CONTEXT


def test_retrieve_context_sentinel_on_empty_results():
    """Empty retriever results → NO_RELEVANT_CONTEXT."""
    retriever = _make_retriever([])

    result = retrieve_context("aspartame", retriever)

    assert result == NO_RELEVANT_CONTEXT


def test_retrieve_context_filters_to_relevant_chunks_only():
    """With two chunks, only the relevant one's content appears in the result."""
    doc_bha = _make_doc("BHA content.", ingredient="bha_bht", aliases="E320, BHA")
    doc_asp = _make_doc("Aspartame content.", ingredient="aspartame", aliases="E951")
    retriever = _make_retriever([doc_bha, doc_asp])

    result = retrieve_context("tell me about aspartame", retriever)

    assert "Aspartame content." in result
    assert "BHA content." not in result
