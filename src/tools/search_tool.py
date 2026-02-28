"""Tavily web-search tool for the multi-agent system.

Tavily is a search engine built specifically for AI agents. Unlike raw Google
Search (which returns HTML snippets that need parsing), Tavily returns
pre-extracted, LLM-optimized content: clean titles, URLs, and text snippets
ready for direct consumption by a language model.

Credit cost model:
    - ``search_depth="basic"``    → 1 API credit per call
    - ``search_depth="advanced"`` → 2 API credits per call

We default to basic depth to conserve the free-tier quota (1,000 credits/month).
"""

from __future__ import annotations

import os

from langchain_core.tools import BaseTool, tool
from langchain_tavily import TavilySearch

from src.config.logging import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


def create_search_tool() -> BaseTool:
    """Create a Tavily search tool configured from application settings.

    Returns a LangChain-compatible tool that wraps ``TavilySearch`` with
    structured logging and graceful error handling.  Errors are returned
    as plain-text strings so the LLM can reason about them and decide
    whether to retry or rephrase the query.

    Returns:
        A LangChain ``BaseTool`` that accepts a search query string.
    """
    _ensure_tavily_env()

    tavily_client = TavilySearch(
        max_results=settings.search_max_results,
        search_depth="basic",
    )

    @tool("tavily_web_search")
    def tavily_web_search(query: str) -> str:
        """Search the web for current information using Tavily.

        Args:
            query: The search query string.

        Returns:
            Search results with titles, URLs, and content snippets,
            or a human-readable error message if the search fails.
        """
        logger.info("tavily_search_started", query=query)

        try:
            results = tavily_client.invoke({"query": query})
        except Exception as exc:
            logger.error("tavily_search_failed", query=query, error=str(exc))
            return (
                f"Search failed: {exc}. "
                "Try rephrasing your query or searching again shortly."
            )

        # TavilySearch returns a list of dicts or a string depending on
        # configuration. Count results for logging.
        result_count = len(results) if isinstance(results, list) else 1
        logger.info(
            "tavily_search_completed",
            query=query,
            result_count=result_count,
        )

        return results

    return tavily_web_search


def create_native_search_tool() -> BaseTool:
    """Return the native ``TavilySearch`` BaseTool without a wrapper.

    The native tool uses the name ``tavily_search`` and a description
    pre-tested with Groq's Llama models.  This avoids a known issue
    where custom ``@tool`` names trigger malformed tool-call generation
    in multi-agent (supervisor) contexts.

    Use this variant inside the swarm; use :func:`create_search_tool`
    for standalone agent usage where the logging wrapper is preferred.
    """
    _ensure_tavily_env()

    return TavilySearch(
        max_results=settings.search_max_results,
        search_depth="basic",
    )


def _ensure_tavily_env() -> None:
    """Bridge pydantic-settings → OS env for TavilySearch's own validator."""
    os.environ.setdefault("TAVILY_API_KEY", settings.tavily_api_key)
