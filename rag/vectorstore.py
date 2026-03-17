"""vectorstore.py — ChromaDB setup and connection.

Provides a single get_vectorstore() factory that returns a persistent
Chroma collection, ready for similarity search.
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


_IS_STREAMLIT_CLOUD = os.path.exists('/mount/src')
CHROMA_PERSIST_DIR = '/tmp/chroma_db' if _IS_STREAMLIT_CLOUD else os.path.join(
    os.path.dirname(__file__), '..', 'knowledge_base', 'data', 'chroma_db'
)
print(f"[vectorstore] CHROMA_PERSIST_DIR = {CHROMA_PERSIST_DIR}")
COLLECTION_NAME: str = "nutrition_kb"

_chroma_client: chromadb.ClientAPI | None = None


def get_chroma_client() -> chromadb.ClientAPI:
    """Return (or create) the shared ChromaDB client.

    Uses EphemeralClient (in-memory) on Streamlit Cloud to avoid filesystem
    schema conflicts between chromadb versions. Uses PersistentClient locally
    so embeddings survive restarts.

    The singleton ensures build_kb and get_vectorstore share the same client
    instance, which is critical on Streamlit Cloud where in-memory data would
    otherwise be lost between calls.
    """
    global _chroma_client
    if _chroma_client is None:
        if _IS_STREAMLIT_CLOUD:
            _chroma_client = chromadb.EphemeralClient()
        else:
            _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _chroma_client


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
        client=get_chroma_client(),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )
