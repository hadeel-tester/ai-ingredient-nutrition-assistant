"""Integration tests for knowledge_base/build_kb.py — build_chroma().

Calls build_chroma() with real .md documents and real OpenAI embeddings.
Verifies that the ChromaDB collection is correctly built, chunk count is correct,
metadata fields are attached, and similarity search returns relevant results.

Prerequisites:
    1. OPENAI_API_KEY set in .env
    2. .md files present in knowledge_base/documents/

Run with:
    pytest tests/knowledge_base/test_build_kb_integration.py -v -s -m integration

Uses a temporary ChromaDB directory — the real knowledge_base/data/chroma_db/
is never modified by these tests.
"""

from __future__ import annotations

import pytest
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from knowledge_base.build_kb import (
    COLLECTION_NAME,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    build_chroma,
    load_md_files,
)

load_dotenv()

# Required metadata fields on every chunk (from _METADATA_FIELDS in build_kb.py)
_REQUIRED_METADATA_FIELDS = {
    "ingredient",
    "category",
    "e_number",
    "aliases",
    "risk_level",
    "eu_status",
    "allergen",
    "vegan",
    "source",  # added by chunk_by_section
}


# ---------------------------------------------------------------------------
# Module-scoped fixture — build once, query many times
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_vectorstore(tmp_path_factory):
    """Build a ChromaDB collection from the real documents into a temp dir.

    scope="module" ensures build_chroma() (and its OpenAI embedding call)
    runs exactly once for all tests in this file.

    Returns:
        Tuple of (Chroma vectorstore, expected_chunk_count).
    """
    persist_dir = tmp_path_factory.mktemp("chroma_integration")

    docs, file_count = load_md_files(DOCUMENTS_DIR)
    assert file_count > 0, (
        f"No .md files found in {DOCUMENTS_DIR}. "
        "Ensure knowledge_base/documents/ contains valid markdown files."
    )

    build_chroma(docs, persist_dir)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    return vs, len(docs)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBuildChromaIntegration:
    """Verify ChromaDB collection built from real .md documents."""

    def test_collection_has_chunks(self, built_vectorstore):
        """Collection must contain at least one chunk after build."""
        vs, _ = built_vectorstore
        count = vs._collection.count()
        assert count > 0, f"Expected chunks in collection, got {count}."

    def test_chunk_count_matches_loaded_documents(self, built_vectorstore):
        """Number of stored chunks must equal chunks produced by load_md_files."""
        vs, expected_count = built_vectorstore
        actual_count = vs._collection.count()
        assert actual_count == expected_count, (
            f"Expected {expected_count} chunks, ChromaDB has {actual_count}."
        )

    def test_each_chunk_has_required_metadata_fields(self, built_vectorstore):
        """Every retrieved chunk must carry all required metadata fields."""
        vs, _ = built_vectorstore
        # Retrieve all stored items including their metadata
        raw = vs._collection.get(include=["metadatas"])
        metadatas: list[dict] = raw["metadatas"]

        assert len(metadatas) > 0, "No metadata returned from collection."

        for i, meta in enumerate(metadatas):
            missing = _REQUIRED_METADATA_FIELDS - set(meta.keys())
            assert not missing, (
                f"Chunk {i} is missing metadata fields: {missing}. Got: {list(meta.keys())}"
            )

    def test_similarity_search_returns_relevant_result(self, built_vectorstore):
        """Searching for 'BHA antioxidant' must return a chunk mentioning BHA or antioxidant."""
        vs, _ = built_vectorstore
        results = vs.similarity_search("BHA antioxidant", k=1)

        assert len(results) > 0, "Similarity search returned no results."

        top_result = results[0]
        content_lower = top_result.page_content.lower()
        assert "bha" in content_lower or "antioxidant" in content_lower, (
            f"Top result does not seem relevant. Content: {top_result.page_content[:200]}"
        )

    def test_retrieved_chunk_has_non_empty_ingredient_metadata(self, built_vectorstore):
        """The 'ingredient' metadata field on retrieved chunks must not be empty."""
        vs, _ = built_vectorstore
        results = vs.similarity_search("BHA antioxidant", k=3)

        for doc in results:
            ingredient = doc.metadata.get("ingredient", "")
            assert ingredient, (
                f"Chunk has empty 'ingredient' metadata. Full metadata: {doc.metadata}"
            )

    def test_similarity_search_different_query(self, built_vectorstore):
        """Searching for 'artificial flavours health risks' must return a result."""
        vs, _ = built_vectorstore
        results = vs.similarity_search("artificial flavours health risks", k=1)

        assert len(results) > 0, "Similarity search for artificial flavours returned no results."
        assert len(results[0].page_content) > 20, "Result content is suspiciously short."
