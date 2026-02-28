"""Benchmark tests for end-to-end agent accuracy.

Each parametrized case mocks the full swarm pipeline and verifies that
the final answer contains expected keywords.  No real API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage


# ---------------------------------------------------------------------------
# Test data: (question, mock_answer, expected_keywords)
# ---------------------------------------------------------------------------

BENCHMARK_CASES = [
    pytest.param(
        "What is the capital of France?",
        "The capital of France is Paris. Paris is known for the Eiffel Tower.",
        ["Paris"],
        id="factual-geography",
    ),
    pytest.param(
        "Explain Python's GIL",
        "The Global Interpreter Lock (GIL) is a mutex that protects access "
        "to Python objects, preventing multiple threads from executing Python "
        "bytecodes at once.",
        ["GIL", "thread"],
        id="technical-concept",
    ),
    pytest.param(
        "What are the benefits of RAG?",
        "Retrieval-Augmented Generation (RAG) improves LLM accuracy by "
        "grounding answers in retrieved documents, reducing hallucinations.",
        ["retrieval", "accuracy"],
        id="ai-concept-rag",
    ),
    pytest.param(
        "Fetch https://api.github.com/repos/langchain-ai/langgraph",
        "LangGraph is a framework for building stateful multi-agent "
        "applications. It has 5000 stars on GitHub.",
        ["LangGraph", "agent"],
        id="url-fetch-github",
    ),
    pytest.param(
        "What are the latest features in Python 3.13?",
        "Python 3.13 introduces a JIT compiler, improved error messages, "
        "and a new REPL. The JIT compiler can boost performance.",
        ["Python", "3.13"],
        id="current-events-python",
    ),
]


class TestBenchmark:
    """Parametrized end-to-end benchmark tests with mocked swarm."""

    @pytest.mark.parametrize("question,mock_answer,expected_keywords", BENCHMARK_CASES)
    @patch("src.swarm.supervisor.create_react_agent")
    @patch("src.swarm.supervisor.create_supervisor")
    @patch("src.swarm.supervisor.create_synthesizer_agent")
    @patch("src.swarm.supervisor.create_url_fetch_tool")
    @patch("src.swarm.supervisor.create_native_search_tool")
    @patch("src.swarm.supervisor.ChatGroq")
    def test_swarm_answer_contains_keywords(
        self,
        mock_groq: MagicMock,
        mock_search_tool: MagicMock,
        mock_url_tool: MagicMock,
        mock_synth_factory: MagicMock,
        mock_create_sup: MagicMock,
        mock_react: MagicMock,
        question: str,
        mock_answer: str,
        expected_keywords: list[str],
    ) -> None:
        """The swarm produces an answer containing the expected keywords."""
        mock_groq.return_value = MagicMock()
        mock_search_tool.return_value = MagicMock()
        mock_url_tool.return_value = MagicMock()
        mock_synth_factory.return_value = MagicMock()
        mock_react.return_value = MagicMock()

        # Build a fake message history with a synthesizer answer
        synth_message = AIMessage(content=mock_answer, name="synthesizer_agent")
        supervisor_message = AIMessage(
            content="The answer has been provided.", name="supervisor"
        )
        fake_result = {
            "messages": [synth_message, supervisor_message],
        }

        # Wire up: create_supervisor() -> workflow.compile() -> app.invoke()
        mock_app = MagicMock()
        mock_app.invoke.return_value = fake_result

        mock_workflow = MagicMock()
        mock_workflow.compile.return_value = mock_app

        mock_create_sup.return_value = mock_workflow

        from src.swarm.supervisor import run_swarm

        # Disable RAG for benchmark tests to avoid embedding model download
        with patch("src.swarm.supervisor.settings") as mock_settings:
            mock_settings.model_name = "llama-3.3-70b-versatile"
            mock_settings.groq_api_key = "test-key"
            mock_settings.rag_enabled = False

            answer = run_swarm(question)

        # Verify answer contains expected keywords (case-insensitive)
        answer_lower = answer.lower()
        for keyword in expected_keywords:
            assert keyword.lower() in answer_lower, (
                f"Expected keyword '{keyword}' not found in answer: {answer[:200]}"
            )
