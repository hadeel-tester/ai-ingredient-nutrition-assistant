"""prompts.py — All system and chain prompts for the nutrition chatbot.

Centralising prompts here makes them easy to iterate without touching
chain logic.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------------------------------------------------------------------------
# System prompt — sets the assistant's persona and behaviour
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """You are a knowledgeable nutrition assistant.
Use the retrieved context below to answer the user's question accurately.
If you cannot find the answer in the context, say so honestly — do not
hallucinate nutritional values.

Context:
{context}
"""

# ---------------------------------------------------------------------------
# RAG chat prompt — used by the main chat chain
# ---------------------------------------------------------------------------

RAG_CHAT_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
