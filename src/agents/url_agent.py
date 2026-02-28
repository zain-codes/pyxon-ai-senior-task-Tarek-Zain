"""URL agent — fetches and analyses web pages and API endpoints.

This module implements a LangGraph ReAct agent that follows the same
Reason → Act → Observe pattern as the search agent (Phase 01), but
with a URL-fetching tool instead of a search tool.

The agent can:
    - Fetch any public URL (HTML page or JSON API).
    - Describe data structures returned by APIs.
    - Summarise article content from web pages.
    - Explain errors (timeouts, 404s, blocked) when fetches fail.

The agent is stateless — conversation memory is handled at the
supervisor / swarm layer (Phase 03).
"""

from __future__ import annotations

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from src.config.logging import get_logger
from src.config.settings import settings
from src.tools.url_fetch_tool import create_url_fetch_tool

logger = get_logger(__name__)

SYSTEM_PROMPT: str = (
    "You are a URL fetching agent that can access any web page or API "
    "endpoint.\n\n"
    "INSTRUCTIONS:\n"
    "1. When given a URL, fetch it and analyze the content.\n"
    "2. For JSON API responses: describe the data structure, list the "
    "fields, and explain what the data represents.\n"
    "3. For HTML web pages: summarize the main content, key points, and "
    "any important data found on the page.\n"
    "4. If a URL fails to load, explain the error clearly (timeout, not "
    "found, blocked, etc.) and suggest possible reasons.\n"
    "5. After completing your analysis, report your findings to the "
    "supervisor.\n\n"
    "RESPONSE FORMAT:\n"
    "Start with a brief summary, then provide details. Include the URL "
    "you fetched as a source reference."
)
"""System prompt that defines the URL agent's behaviour and output format."""


def create_url_agent():
    """Create a LangGraph ReAct agent wired to the URL fetch tool.

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

    url_tool = create_url_fetch_tool()

    agent = create_react_agent(
        model=model,
        tools=[url_tool],
        name="url_agent",
        prompt=SYSTEM_PROMPT,
    )

    logger.info("url_agent_created", model=settings.model_name)
    return agent


def run_url_agent(question: str) -> str:
    """Run a single question through the URL agent and return the answer.

    Creates a fresh agent per call (stateless).  Handles Groq rate limits
    and general exceptions gracefully.

    Args:
        question: The user's natural-language question (should contain a URL).

    Returns:
        The agent's analysis as a string.
        On failure, returns a user-friendly error message.
    """
    logger.info("url_agent_started", question=question)

    try:
        agent = create_url_agent()
        result = agent.invoke({"messages": [("user", question)]})
        answer = result["messages"][-1].content

        logger.info("url_agent_finished", answer_length=len(answer))
        return answer

    except Exception as exc:
        error_name = type(exc).__name__

        if "429" in str(exc) or "rate" in str(exc).lower():
            logger.warning("url_agent_rate_limited", error=str(exc))
            return (
                "The LLM is temporarily rate-limited. "
                "Please wait a moment and try again."
            )

        logger.error("url_agent_failed", error=error_name, detail=str(exc))
        return f"An error occurred while processing your question: {exc}"
