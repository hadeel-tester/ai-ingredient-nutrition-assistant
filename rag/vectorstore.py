"""vectorstore.py — ChromaDB setup and connection.

Provides a single get_vectorstore() factory that returns a persistent
Chroma collection, ready for similarity search.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME: str = "nutrition_kb"


def get_embeddings() -> OpenAIEmbeddings:
    """Return the configured OpenAI embedding model.

    Returns:
        OpenAIEmbeddings instance using text-embedding-ada-002.
    """
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_vectorstore() -> Chroma:
    """Return (or create) the persistent ChromaDB vector store.

    The store is persisted at CHROMA_PERSIST_DIR so embeddings survive
    restarts.  Run knowledge_base/build_kb.py first to populate it.

    Returns:
        Chroma instance connected to the local collection.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
