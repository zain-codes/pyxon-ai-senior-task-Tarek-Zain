"""Synthesizer agent — combines multi-source outputs into a coherent answer.

This agent has NO tools.  It is a pure writer that receives research
findings collected by the search and URL agents, then produces a
polished, well-cited final answer suitable for enterprise clients.

It is only invoked as part of the supervisor swarm (never standalone).
"""

from __future__ import annotations

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from src.config.logging import get_logger
from src.config.settings import settings

logger = get_logger(__name__)

SYSTEM_PROMPT: str = (
    "You are a synthesis agent. Your job is to take research findings from "
    "other agents and produce a clear, well-structured final answer.\n\n"
    "INSTRUCTIONS:\n"
    "1. You will receive research data collected by other agents in the "
    "conversation.\n"
    "2. Organize the information logically with clear structure.\n"
    "3. Remove redundant or contradictory information.\n"
    "4. Always include citations: [Source Title](URL) for every major "
    "claim.\n"
    "5. If the research is insufficient to fully answer the question, say "
    "so and explain what additional information would be needed.\n"
    "6. Write in a professional, concise tone suitable for enterprise "
    "clients.\n\n"
    "RESPONSE FORMAT:\n"
    "## Answer\n"
    "<Clear, structured response>\n\n"
    "## Sources\n"
    "- [Title](URL)\n"
    "- [Title](URL)"
)
"""System prompt that defines the synthesizer's behaviour and output format."""


def create_synthesizer_agent():
    """Create a tool-less LangGraph ReAct agent for answer synthesis.

    The synthesizer uses the same Groq-hosted LLM as the other agents
    but carries no tools — it only writes.

    Returns:
        A compiled LangGraph ``CompiledStateGraph`` that accepts
        conversation messages and produces a polished final answer.
    """
    model = ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        temperature=0,
    )

    agent = create_react_agent(
        model=model,
        tools=[],
        name="synthesizer_agent",
        prompt=SYSTEM_PROMPT,
    )

    logger.info("synthesizer_agent_created", model=settings.model_name)
    return agent
