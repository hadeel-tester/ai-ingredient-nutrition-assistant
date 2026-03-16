"""token_tracker.py — Session-level token usage and cost tracker.

Accumulates input/output token counts across all LangChain calls in a session
and calculates cost using OpenAI gpt-4o-mini pricing.

Usage:
    tracker = get_tracker()      # retrieves or creates from st.session_state
    tracker.record(usage_dict)   # call after each chain invoke
    stats = tracker.get_cost()   # returns totals + formatted cost
"""

from __future__ import annotations

import streamlit as st

# OpenAI gpt-4o-mini pricing (USD per 1M tokens, as of 2024)
_PRICE_INPUT_PER_M: float = 0.15
_PRICE_OUTPUT_PER_M: float = 0.60

_SESSION_KEY = "token_tracker"


class TokenTracker:
    """Accumulates token usage across LangChain calls and calculates cost.

    Attributes:
        input_tokens:  Total input (prompt) tokens recorded this session.
        output_tokens: Total output (completion) tokens recorded this session.
    """

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def record(self, usage: dict) -> None:
        """Accumulate token counts from a LangChain usage dict.

        Accepts the usage metadata dict returned by LangChain when
        stream_usage=True is set on ChatOpenAI. Keys may be
        'input_tokens' / 'output_tokens' (LangChain) or
        'prompt_tokens' / 'completion_tokens' (raw OpenAI).
        Both formats are handled gracefully.

        Args:
            usage: Dict containing token count fields.
        """
        self.input_tokens += (
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or 0
        )
        self.output_tokens += (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or 0
        )

    def get_cost(self) -> dict:
        """Calculate cumulative cost using gpt-4o-mini pricing.

        Returns:
            Dict with keys:
              - input_tokens (int)
              - output_tokens (int)
              - total_tokens (int)
              - cost_usd (float): raw cost in USD
              - cost_formatted (str): e.g. '$0.0023'
        """
        cost_usd = (
            self.input_tokens  / 1_000_000 * _PRICE_INPUT_PER_M
            + self.output_tokens / 1_000_000 * _PRICE_OUTPUT_PER_M
        )
        return {
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens":  self.input_tokens + self.output_tokens,
            "cost_usd":      cost_usd,
            "cost_formatted": f"${cost_usd:.4f}",
        }

    def reset(self) -> None:
        """Reset all accumulated token counts to zero."""
        self.input_tokens = 0
        self.output_tokens = 0


def get_tracker() -> TokenTracker:
    """Return the session-level TokenTracker, creating it if not present.

    Stores the tracker in st.session_state so it persists across Streamlit
    reruns within the same browser session.

    Returns:
        The TokenTracker instance for the current session.
    """
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = TokenTracker()
    return st.session_state[_SESSION_KEY]
