"""retriever.py — Query translation and retrieval logic.

Wraps the vector store in a LangChain retriever with optional
query rewriting before similarity search.

Relevance checking:
    retrieve_context() compares each chunk's `ingredient` and `aliases`
    metadata against the user's query before passing context to the LLM.
    If no chunk matches, it returns NO_RELEVANT_CONTEXT so the LLM knows
    to fall back to general knowledge with a ⚠️ disclosure warning.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag.vectorstore import get_vectorstore

DEFAULT_K: int = 4  # number of chunks to retrieve per query

# Sentinel injected as {context} when no retrieved chunk matches the query.
# The system prompt checks for this exact string to trigger the ⚠️ warning.
NO_RELEVANT_CONTEXT: str = "NO_RELEVANT_CONTEXT_FOUND"


def get_retriever(k: int = DEFAULT_K) -> BaseRetriever:
    """Return a retriever over the nutrition knowledge base.

    Args:
        k: Number of top documents to return per query.

    Returns:
        A LangChain retriever ready for use in a chain.
    """
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def _matches_query(metadata: dict, query_lower: str) -> bool:
    """Return True if the chunk's ingredient name or any alias appears in the query.

    Matching is one-directional: we check whether the known ingredient name or
    alias is a substring of the user's query (case-insensitive). This prevents
    false positives — e.g. "BHA" matching a query about "baharat spice" would
    require the full alias "bha" to appear in the query as a substring.

    Underscore normalisation: ingredient names stored as "artificial_flavours"
    are converted to "artificial flavours" before matching so that natural
    language queries ("artificial flavours risks") are found correctly.

    Args:
        metadata:    Metadata dict from a LangChain Document.
        query_lower: The user's query, already lowercased.

    Returns:
        True if at least one candidate (ingredient name or alias) is found
        as a substring in query_lower.
    """
    ingredient = metadata.get("ingredient", "").replace("_", " ").lower()
    aliases_raw = metadata.get("aliases", "")
    aliases = [a.strip().lower() for a in aliases_raw.split(",") if a.strip()]

    candidates = [ingredient] + aliases
    return any(c and c in query_lower for c in candidates)


def retrieve_context(query: str, retriever: BaseRetriever) -> str:
    """Retrieve chunks and return formatted context or NO_RELEVANT_CONTEXT.

    Calls the retriever, then filters the results to only those whose
    `ingredient` or `aliases` metadata matches the user's query. If at least
    one chunk matches, returns the joined page content for the LLM. If none
    match, returns the NO_RELEVANT_CONTEXT sentinel so the LLM knows to apply
    the general-knowledge fallback with the ⚠️ disclosure warning.

    Args:
        query:     The user's input string.
        retriever: A LangChain retriever over the ChromaDB knowledge base.

    Returns:
        Formatted context string, or NO_RELEVANT_CONTEXT if no chunk is relevant.
    """
    docs = retriever.invoke(query)
    query_lower = query.lower()
    relevant = [doc for doc in docs if _matches_query(doc.metadata, query_lower)]
    if not relevant:
        return NO_RELEVANT_CONTEXT
    return "\n\n".join(doc.page_content for doc in relevant)


def get_relevant_sources(query: str, retriever: BaseRetriever) -> list[str]:
    """Return ingredient names for chunks whose metadata matches the query.

    Used by the UI to populate the Sources expander without a separate LLM call.

    Args:
        query:     The user's input string.
        retriever: A LangChain retriever over the ChromaDB knowledge base.

    Returns:
        Deduplicated list of ingredient name strings (e.g. ["aspartame"]).
    """
    docs = retriever.invoke(query)
    query_lower = query.lower()
    seen: list[str] = []
    for doc in docs:
        if _matches_query(doc.metadata, query_lower):
            name = doc.metadata.get("ingredient", "")
            if name and name not in seen:
                seen.append(name)
    return seen


def retrieve_with_scores(
    query: str,
    vectorstore: Chroma,
    k: int = DEFAULT_K,
) -> list[tuple[Document, float, bool]]:
    """Return (doc, similarity_score, is_relevant) for each retrieved chunk.

    Calls vectorstore.similarity_search_with_score() directly to expose the
    similarity score — the standard retriever interface does not return scores.

    ChromaDB returns cosine *distance* in [0, 2] (0 = identical, 2 = opposite).
    This function normalises to a [0, 1] similarity: similarity = 1 - distance/2.

    Args:
        query:       User input string.
        vectorstore: Chroma instance (must be the same collection as the retriever).
        k:           Number of chunks to retrieve.

    Returns:
        List of (Document, similarity_score, is_relevant) tuples, ordered by
        descending similarity. is_relevant mirrors the _matches_query filter used
        by retrieve_context() so the UI can distinguish used vs filtered-out chunks.
    """
    pairs = vectorstore.similarity_search_with_score(query, k=k)
    query_lower = query.lower()
    result: list[tuple[Document, float, bool]] = []
    for doc, distance in pairs:
        similarity = max(0.0, 1.0 - (distance / 2))
        is_relevant = _matches_query(doc.metadata, query_lower)
        result.append((doc, similarity, is_relevant))
    return result
