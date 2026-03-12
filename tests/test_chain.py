"""Integration tests for chains/chat_chain.py.

Sends 3 real messages through the full agent chain (RAG + OpenAI + tools).
Prints the complete response and which tools were called for each message.

Prerequisites:
    1. OPENAI_API_KEY set in .env
    2. ChromaDB populated: python -m knowledge_base.build_kb

Run with:
    pytest tests/test_chain.py -v -s -m integration

The -s flag is required to see print() output in the terminal.
"""

from __future__ import annotations

import pytest

from chains.chat_chain import build_chat_chain

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEPARATOR = "-" * 60


def _invoke_and_report(chain, message: str) -> tuple[str, list[str]]:
    """Invoke the chain, print the response and tools called, return both.

    Args:
        chain:   The built agent chain from build_chat_chain().
        message: User question to send.

    Returns:
        Tuple of (response_text, tools_called_list).
    """
    result = chain.invoke({"input": message, "chat_history": []})
    response: str = result["output"]
    tools_called: list[str] = [
        step[0].tool for step in result.get("intermediate_steps", [])
    ]

    print(_SEPARATOR)
    print(f"Q: {message}")
    print(f"A: {response}")
    print(f"Tools called: {tools_called if tools_called else ['none']}")
    print(_SEPARATOR)

    return response, tools_called


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestChatChain:
    """End-to-end tests for the RAG + tool-calling agent chain.

    The chain fixture is scoped to the class so it is built once across all
    three tests — connecting to ChromaDB and initialising the LLM is expensive.
    """

    @pytest.fixture(scope="class")
    def chain(self):
        """Build the agent chain once for the entire test class."""
        return build_chat_chain()

    def test_rag_question_returns_non_empty_response(self, chain):
        """A knowledge-base question should return a substantive response.

        'Why is palm oil bad for you?' is covered by the RAG knowledge base
        (artificial_flavours.md / BHA_BHT.md) and should not need a tool call.
        """
        message = "Why is palm oil bad for you?"
        response, tools_called = _invoke_and_report(chain, message)

        assert len(response) > 50, "Response should be substantive, not empty or trivial."
        assert "error" not in response.lower(), "Response should not be an error message."

    def test_allergen_question_calls_check_allergens(self, chain):
        """An allergen-detection question should trigger the check_allergens tool.

        The LLM is prompted to use tools for allergen checks on ingredient lists.
        """
        message = (
            "Does this product contain allergens: wheat flour, eggs, milk powder"
        )
        response, tools_called = _invoke_and_report(chain, message)

        assert len(response) > 0, "Response must not be empty."
        assert "check_allergens" in tools_called, (
            f"Expected check_allergens to be called, got: {tools_called}"
        )

    def test_health_score_question_calls_score_health(self, chain):
        """A health scoring question should trigger the score_health tool.

        The LLM is prompted to use the tool when the user asks for a health score.
        """
        message = (
            "Give me a health score for a product with these nutrients: "
            "calories 450, sugar 20g, fat 15g, salt 1.2g per 100g. "
            "Ingredients: wheat flour, palm oil, sugar, salt, E471, E320."
        )
        response, tools_called = _invoke_and_report(chain, message)

        assert len(response) > 0, "Response must not be empty."
        assert "score_health" in tools_called, (
            f"Expected score_health to be called, got: {tools_called}"
        )
