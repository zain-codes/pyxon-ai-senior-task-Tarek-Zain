"""Search agent — finds current information from the web using Tavily + Groq.

This module implements a LangGraph ReAct agent that follows the
Reason → Act → Observe loop:

    1. The LLM **reasons** about the user's question.
    2. It **acts** by calling the Tavily search tool.
    3. It **observes** the search results.
    4. It repeats (up to 3 searches) until it has enough information.
    5. It produces a final answer with ``[Source](URL)`` citations.

The agent is stateless — each call creates a fresh conversation with no
memory of prior queries.  This simplicity is intentional; conversation
memory is handled at the supervisor / swarm layer (Phase 03).
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

from src.config.logging import get_logger
from src.config.settings import settings
from src.tools.search_tool import create_search_tool

logger = get_logger(__name__)

SYSTEM_PROMPT: str = (
    "You are a research agent specialized in finding current information "
    "from the web.\n\n"
    "INSTRUCTIONS:\n"
    "1. Use the search tool to find relevant, up-to-date information.\n"
    "2. If the first search doesn't give enough information, refine your "
    "query and search again (up to 3 searches maximum).\n"
    "3. Always ground your answer in the search results — do not make "
    "things up.\n"
    "4. Include citations as [Source Title](URL) for every claim you make.\n"
    "5. If you cannot find relevant information, say so honestly.\n"
    "6. After completing your research, report your findings to the "
    "supervisor.\n\n"
    "RESPONSE FORMAT:\n"
    "Provide a clear, well-structured answer followed by a Sources section "
    "listing all URLs you referenced."
)
"""System prompt that defines the search agent's behaviour and output format."""


def create_search_agent():
    """Create a LangGraph ReAct agent wired to the Tavily search tool.

    The agent uses a Groq-hosted LLM (``settings.model_name``) with
    ``temperature=0`` for deterministic, reproducible output.

    Returns:
        A compiled LangGraph ``CompiledStateGraph`` ready to be invoked
        with ``{"messages": [("user", "<question>")]}``.
    """
    model = ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        temperature=0,
    )

    search_tool = create_search_tool()

    agent = create_react_agent(
        model=model,
        tools=[search_tool],
        name="research_agent",
        prompt=SYSTEM_PROMPT,
    )

    logger.info("search_agent_created", model=settings.model_name)
    return agent


def run_search_agent(question: str) -> str:
    """Run a single question through the search agent and return the answer.

    This is a convenience wrapper for standalone testing and for use in
    ``main.py``.  It creates a fresh agent, invokes it, and extracts the
    final message content.

    Args:
        question: The user's natural-language question.

    Returns:
        The agent's final answer as a string, including citations.
        On failure, returns a user-friendly error message.
    """
    logger.info("search_agent_started", question=question)

    try:
        agent = create_search_agent()
        result = agent.invoke({"messages": [("user", question)]})
        answer = result["messages"][-1].content

        logger.info("search_agent_finished", answer_length=len(answer))
        return answer

    except Exception as exc:
        error_name = type(exc).__name__

        # Groq free-tier rate limits surface as 429 responses.
        if "429" in str(exc) or "rate" in str(exc).lower():
            logger.warning("search_agent_rate_limited", error=str(exc))
            return (
                "The LLM is temporarily rate-limited. "
                "Please wait a moment and try again."
            )

        logger.error("search_agent_failed", error=error_name, detail=str(exc))
        return f"An error occurred while processing your question: {exc}"
