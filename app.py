"""NutriMind — Streamlit entry point.

Initialises all expensive resources once (LangChain agent + ChromaDB vectorstore)
using st.cache_resource, then delegates rendering to the two page modules:
  - ui/chat.py           — RAG chatbot with tool calling
  - ui/knowledge_base.py — ingredient card grid

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="NutriMind",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading NutriMind — connecting to knowledge base...")
def _load_resources():
    """Load the chain and vectorstore once per app session.

    st.cache_resource persists the return value across all reruns (button
    clicks, new messages) so ChromaDB and the LLM client are initialised only
    once, not on every Streamlit interaction.

    Both get_vectorstore() and build_kb() are called inside this cached
    function so they share the same _chroma_client singleton from
    rag.vectorstore. Calling build_kb() outside this scope (e.g. via a
    separate _ensure_kb_built()) would create the EphemeralClient in a
    different execution context, leaving the vectorstore pointing at an
    empty in-memory instance on Streamlit Cloud.
    """
    from chains.chat_chain import build_chat_chain
    from rag.vectorstore import get_vectorstore
    from knowledge_base.build_kb import main as build_kb

    chain = build_chat_chain()
    vectorstore = get_vectorstore()
    if vectorstore._collection.count() == 0:
        with st.spinner('Building knowledge base — first-time setup, ~30 seconds...'):
            build_kb()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return chain, vectorstore, retriever


chain, vectorstore, retriever = _load_resources()

# Load profile on every run (uncached) so it always reflects the latest save.
from ui.profile import load_user_profile  # noqa: E402 — after st.set_page_config
user_profile = load_user_profile()


def _chat_page() -> None:
    from ui.chat import render_chat_page
    render_chat_page(chain, vectorstore, retriever, user_profile)


def _kb_page() -> None:
    from ui.knowledge_base import render_kb_page
    render_kb_page(vectorstore)


def _profile_page() -> None:
    from ui.profile import render_profile_page
    render_profile_page()


pg = st.navigation(
    [
        st.Page(_chat_page,    title="Chat",           icon="💬", default=True),
        st.Page(_kb_page,      title="Knowledge Base", icon="📚"),
        st.Page(_profile_page, title="Profile",        icon="👤"),
    ]
)
pg.run()
