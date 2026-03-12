"""chat.py — Main Streamlit chat page.

Renders the full conversational UI:
  - Sidebar: app info, tool list (labelled "run automatically"), barcode lookup, clear button
  - Message history (user right / assistant left)
  - Step-by-step progress via st.status() + background thread + stop button
  - Per-assistant-message expanders: Sources, Tools used, RAG process
  - ⚠️ general-knowledge warning rendered as st.warning() (yellow box)

Stop button design:
    chain.invoke() is synchronous and blocking. To allow cancellation, it runs
    in a background threading.Thread. The main script polls a shared mutable
    result_container dict (GIL-safe for simple assignments) and reruns every
    0.4 s until the thread completes or the user clicks Stop.
"""

from __future__ import annotations

import threading
import time

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever

from rag.retriever import retrieve_with_scores
from ui.components import rag_process_expander, tool_result_card

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WARNING_MARKER = "⚠️"

_TOOL_DISPLAY: dict[str, tuple[str, str]] = {
    "lookup_product":               ("🔍", "Open Food Facts"),
    "analyze_product_ingredients":  ("🧬", "Ingredient Analyzer"),
    "check_allergens":              ("⚠️", "Allergen Checker"),
    "evaluate_nutrition":           ("📊", "Nutrition Calculator"),
    "score_health":                 ("🏅", "Health Scorer"),
}

_SIDEBAR_TOOLS = [
    ("🔍", "Open Food Facts", "Live product lookup by name or barcode"),
    (
        "🧬",
        "Ingredient Analyzer",
        "Cross-references each ingredient against the knowledge base "
        "for health risks, safety data, and regulatory status",
    ),
    ("⚠️", "Allergen Checker", "Detects EU-regulated allergens in an ingredient list"),
    (
        "📊",
        "Nutrition Calculator",
        "Objective analysis: per-serving amounts, % of daily intake (DRI), "
        "and green/amber/red traffic lights per nutrient",
    ),
    (
        "🏅",
        "Health Scorer",
        "Overall verdict: 0–100 score + A–F grade weighing nutrients "
        "AND ingredients (additives, palm oil, E-numbers)",
    ),
]

# ---------------------------------------------------------------------------
# Background thread helper
# ---------------------------------------------------------------------------


def _run_chain(chain, input_dict: dict, container: dict) -> None:
    """Invoke the chain in a background thread and write result into container.

    container is a plain dict shared between this thread and the Streamlit
    main thread. Python's GIL makes simple dict key assignments atomic, so
    no explicit locking is needed.

    Args:
        chain:      The built LangChain agent.
        input_dict: Keys "input" and "chat_history" for chain.invoke().
        container:  Shared dict; fields set here: status, result, error.
    """
    try:
        container["result"] = chain.invoke(input_dict)
        container["status"] = "done"
    except Exception as exc:  # noqa: BLE001 — surface all errors to UI
        container["error"] = str(exc)
        container["status"] = "error"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _render_assistant_text(text: str) -> None:
    """Render assistant response, splitting out ⚠️ lines as st.warning boxes.

    The LLM appends a ⚠️ disclaimer when it answers from general knowledge.
    This function separates the main response from that disclaimer so the
    yellow warning box is visually distinct from the regular markdown text.

    Args:
        text: Full assistant response string.
    """
    if _WARNING_MARKER not in text:
        st.markdown(text)
        return
    parts = text.split(_WARNING_MARKER, maxsplit=1)
    main_text = parts[0].rstrip()
    if main_text:
        st.markdown(main_text)
    st.warning(_WARNING_MARKER + parts[1].strip())


def _render_message(msg: dict) -> None:
    """Render a single assistant message with all its expanders.

    Args:
        msg: Message dict from st.session_state.messages.
    """
    _render_assistant_text(msg["content"])

    sources: list[str] = msg.get("sources", [])
    if sources:
        with st.expander("📚 Sources from knowledge base", expanded=False):
            for src in sources:
                st.markdown(f"- **{src.replace('_', ' ').title()}**")

    tools_called: list[dict] = msg.get("tools_called", [])
    if tools_called:
        with st.expander(f"🔧 Tools used ({len(tools_called)})", expanded=False):
            for tool in tools_called:
                icon, display_name = _TOOL_DISPLAY.get(
                    tool["name"], ("🔧", tool["name"])
                )
                tool_result_card(f"{icon} {display_name}", tool["output"])

    rag_chunks: list[dict] = msg.get("rag_chunks", [])
    rag_process_expander(rag_chunks)


def _build_chat_history(messages: list[dict]) -> list:
    """Convert session state dicts to LangChain BaseMessage objects.

    Args:
        messages: List of message dicts from st.session_state.messages.

    Returns:
        List of HumanMessage / AIMessage instances.
    """
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


def _extract_tools(intermediate_steps: list) -> list[dict]:
    """Extract tool call details from AgentExecutor intermediate steps.

    Args:
        intermediate_steps: result["intermediate_steps"] from chain.invoke().

    Returns:
        List of dicts with keys: name, input, output.
    """
    tools = []
    for action, output in intermediate_steps:
        tools.append(
            {
                "name": action.tool,
                "input": action.tool_input,
                "output": output,
            }
        )
    return tools


def _render_sidebar() -> None:
    """Render the sidebar: app info, tools, barcode lookup, clear button."""
    with st.sidebar:
        st.title("🥦 NutriMind")
        st.caption(
            "AI-powered ingredient and nutrition assistant. "
            "Ask about ingredients, allergens, nutritional values, or scan a product."
        )
        st.divider()

        st.markdown("**Quick product lookup**")
        lookup_mode = st.radio(
            "Lookup mode",
            options=["By name", "By barcode"],
            horizontal=True,
            key="lookup_mode",
            label_visibility="collapsed",
        )

        if lookup_mode == "By name":
            product_name = st.text_input(
                "Enter product name",
                placeholder="e.g. Nutella, Coca-Cola",
                key="product_name_input",
                label_visibility="collapsed",
            )
            if st.button("🔍 Look up", use_container_width=True) and product_name.strip():
                st.session_state.barcode_query = f"Look up product: {product_name.strip()}"
                st.rerun()
        else:
            barcode = st.text_input(
                "Enter barcode",
                placeholder="e.g. 3017620422003",
                key="barcode_input",
                label_visibility="collapsed",
            )
            if st.button("🔍 Look up", use_container_width=True) and barcode.strip():
                st.session_state.barcode_query = barcode.strip()
                st.rerun()

        st.divider()

        st.markdown("**AI tools** _(run automatically from chat)_")
        st.caption("The assistant uses these tools on its own — no need to click anything.")
        for icon, name, description in _SIDEBAR_TOOLS:
            st.markdown(f"{icon} **{name}**  \n_{description}_")

        st.divider()

        if st.button("🗑️ Clear conversation", use_container_width=True):
            # Reset all chain state alongside messages
            st.session_state.messages = []
            st.session_state.chain_running = False
            st.session_state.chain_cancelled = False
            st.session_state.chain_result_container = {}
            st.rerun()


def _start_chain(
    chain,
    prompt: str,
    vectorstore: Chroma,
    messages_snapshot: list[dict],
    user_profile: dict,
) -> None:
    """Kick off the background chain thread and store state in session_state.

    This function:
      1. Runs retrieve_with_scores() synchronously (fast, ~0.1 s)
      2. Starts the chain.invoke() thread
      3. Stores pending state in session_state for the polling loop

    Args:
        chain:             The built LangChain agent.
        prompt:            Current user input.
        vectorstore:       Chroma instance for similarity search.
        messages_snapshot: Messages list at submit time (excludes current prompt).
        user_profile:      User profile dict loaded from data/user_profile.json.
    """
    # Step 1 — KB retrieval (fast, run in main thread so st.status can show it)
    scored_chunks = retrieve_with_scores(prompt, vectorstore)
    rag_chunks = [
        {
            "ingredient": doc.metadata.get("ingredient", "?"),
            "section": doc.metadata.get("section", "?"),
            "score": score,
            "is_relevant": is_rel,
        }
        for doc, score, is_rel in scored_chunks
    ]

    chat_history = _build_chat_history(messages_snapshot)
    container: dict = {"status": "running", "result": None, "error": None}

    thread = threading.Thread(
        target=_run_chain,
        args=(
            chain,
            {"input": prompt, "chat_history": chat_history, "user_profile": user_profile},
            container,
        ),
        daemon=True,
    )

    st.session_state.chain_running = True
    st.session_state.chain_cancelled = False
    st.session_state.chain_result_container = container
    st.session_state.chain_thread = thread
    st.session_state.pending_prompt = prompt
    st.session_state.pending_rag_chunks = rag_chunks

    thread.start()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_chat_page(
    chain,
    vectorstore: Chroma,
    retriever: BaseRetriever,
    user_profile: dict,
) -> None:
    """Render the main chat interface.

    Args:
        chain:        The built LangChain agent (from build_chat_chain()).
        vectorstore:  Chroma instance for RAG scores + KB page.
        retriever:    LangChain retriever (passed for type consistency; chain
                      uses its own retriever internally).
        user_profile: User profile dict loaded from data/user_profile.json.
                      Pass {} if no profile is saved yet.
    """
    _render_sidebar()

    # Initialise all session state keys
    for key, default in [
        ("messages", []),
        ("chain_running", False),
        ("chain_cancelled", False),
        ("chain_result_container", {}),
        ("chain_thread", None),
        ("pending_prompt", ""),
        ("pending_rag_chunks", []),
        ("barcode_query", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.header("💬 Chat")
    st.caption(
        "Ask me about ingredients, allergens, nutritional values, or product health scores."
    )

    # Display existing message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_message(msg)
            else:
                st.markdown(msg["content"])

    # -----------------------------------------------------------------------
    # Polling loop — runs on every rerun while chain is in flight
    # -----------------------------------------------------------------------
    if st.session_state.chain_running:
        prompt = st.session_state.pending_prompt
        container = st.session_state.chain_result_container

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            stop_placeholder = st.empty()

            with status_placeholder.container():
                st.status("🤖 AI agent is thinking...", expanded=True)

            if stop_placeholder.button("⏹ Stop", key="stop_btn"):
                st.session_state.chain_running = False
                st.session_state.chain_cancelled = True
                # Remove the pending user message we added earlier
                if (st.session_state.messages
                        and st.session_state.messages[-1]["role"] == "user"):
                    st.session_state.messages.pop()
                status_placeholder.empty()
                stop_placeholder.empty()
                st.info("Request cancelled.")
                st.rerun()
                return

            if container.get("status") == "running":
                time.sleep(0.4)
                st.rerun()
                return

            # Thread completed — clear placeholders and render result
            status_placeholder.empty()
            stop_placeholder.empty()
            st.session_state.chain_running = False

            if container.get("status") == "error":
                st.error(f"Error: {container['error']}")
                if (st.session_state.messages
                        and st.session_state.messages[-1]["role"] == "user"):
                    st.session_state.messages.pop()
                st.rerun()
                return

            result = container["result"]
            response_text = result["output"]
            tools_called = _extract_tools(result.get("intermediate_steps", []))
            rag_chunks = st.session_state.pending_rag_chunks
            sources = [c["ingredient"] for c in rag_chunks if c["is_relevant"]]

            assistant_msg: dict = {
                "role": "assistant",
                "content": response_text,
                "sources": sources,
                "tools_called": tools_called,
                "rag_chunks": rag_chunks,
            }
            _render_message(assistant_msg)
            st.session_state.messages.append(assistant_msg)
        st.rerun()

    # -----------------------------------------------------------------------
    # New prompt handling — chat input or barcode shortcut
    # -----------------------------------------------------------------------

    # Check for barcode submitted from sidebar
    prompt: str = ""
    if st.session_state.get("barcode_query"):
        prompt = st.session_state.barcode_query
        st.session_state.barcode_query = ""

    # Check for chat input
    chat_prompt = st.chat_input(
        "e.g. Is aspartame safe? / Check allergens in: wheat, milk, soy"
    )
    if chat_prompt:
        prompt = chat_prompt

    if prompt:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # KB retrieval + start thread (with progress steps)
        with st.chat_message("assistant"):
            with st.status("Searching knowledge base...", expanded=True) as status:
                status.write("🔍 Searching knowledge base...")
                # Pass a snapshot excluding the just-added user message as history
                messages_before = st.session_state.messages[:-1]
                _start_chain(chain, prompt, vectorstore, messages_before, user_profile)
                status.write("🤖 AI agent started — processing your request...")
                status.update(label="AI is thinking...", state="running")

        st.rerun()
