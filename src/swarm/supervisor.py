"""Supervisor — LangGraph multi-agent orchestrator.

Implements the supervisor pattern: a manager agent receives user queries,
routes them to specialist worker agents (research, URL fetch, synthesizer),
and coordinates the flow until a polished answer is produced.

The supervisor never answers questions itself — it delegates to the right
agent and lets the synthesizer write the final response.

When ``settings.rag_enabled`` is True, agent findings are ingested into a
FAISS vector store after the swarm completes.  The most relevant chunks are
retrieved and fed to the synthesizer for a second, better-grounded pass.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from src.agents.synthesizer import create_synthesizer_agent
from src.config.logging import get_logger
from src.config.settings import settings
from src.tools.search_tool import create_native_search_tool
from src.tools.url_fetch_tool import create_url_fetch_tool

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Sub-agent prompts (supervisor-aware versions of the Phase 01/02 prompts)
# ---------------------------------------------------------------------------

RESEARCH_PROMPT: str = (
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
    "supervisor."
)

URL_PROMPT: str = (
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
    "supervisor."
)

SUPERVISOR_PROMPT: str = (
    "You are a supervisor managing a team of 3 specialized AI agents.\n\n"
    "TEAM:\n"
    "1. research_agent: Can search the web using Tavily. Use for general "
    "questions, current events, finding information about topics.\n"
    "2. url_agent: Can fetch specific URLs and APIs. Use when the user "
    "provides a specific URL, asks about a webpage, or wants API data.\n"
    "3. synthesizer_agent: Writes the final polished answer. Has no tools. "
    "Always use this agent LAST after research/fetching is complete.\n\n"
    "ROUTING RULES:\n"
    "1. If the question is about general knowledge or current events → "
    "route to research_agent first.\n"
    "2. If the question mentions a specific URL or API endpoint → "
    "route to url_agent.\n"
    "3. If the question requires BOTH web search AND specific URL data → "
    "route to research_agent first, then url_agent.\n"
    "4. ALWAYS route to synthesizer_agent as the FINAL step to produce "
    "the polished answer from all collected data.\n"
    "5. If an agent's response is insufficient, you may route back to "
    "the same agent with a refined request.\n\n"
    "IMPORTANT: Do not try to answer the question yourself. Your job is "
    "to delegate to the right agents and let the synthesizer write the "
    "final answer."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_agent_findings(messages: list[BaseMessage]) -> list[str]:
    """Extract substantive text from research_agent and url_agent messages."""
    findings: list[str] = []
    for msg in messages:
        if (
            isinstance(msg, AIMessage)
            and getattr(msg, "name", None) in ("research_agent", "url_agent")
            and msg.content
            and len(msg.content) > 50
            and not msg.content.startswith("Transferring")
        ):
            findings.append(msg.content)
    return findings


def _rag_enhanced_synthesis(
    question: str,
    findings: list[str],
) -> str:
    """Ingest agent findings into RAG, retrieve relevant chunks, synthesize.

    Imports RAGStore lazily so the embedding model is never loaded when
    ``rag_enabled`` is False.
    """
    from src.rag.vectorstore import RAGStore

    store = RAGStore()
    chunk_count = store.ingest(findings)
    logger.info("rag_ingest_complete", chunk_count=chunk_count)

    chunks = store.retrieve(question, k=3)
    logger.info("rag_retrieve_complete", chunk_count=len(chunks))
    store.clear()

    if not chunks:
        return ""

    context = "\n\n---\n\n".join(chunks)
    return (
        "Here are the most relevant excerpts from the collected research:\n\n"
        f"{context}\n\n"
        "Use these to write your answer."
    )


def _extract_answer(messages: list[BaseMessage]) -> str:
    """Find the synthesizer's answer in the message history.

    The supervisor's final message is typically a short wrapper like
    "The answer has been provided."  We want the synthesizer's actual
    content instead — the last substantive ``AIMessage`` from an agent
    (preferring ``synthesizer_agent``).
    """
    # Walk backwards: prefer synthesizer, then any agent with real content.
    for msg in reversed(messages):
        if (
            isinstance(msg, AIMessage)
            and getattr(msg, "name", None) == "synthesizer_agent"
            and msg.content
            and not msg.content.startswith("Transferring")
        ):
            return msg.content

    # Fallback: last AIMessage with substantial content from any agent.
    for msg in reversed(messages):
        if (
            isinstance(msg, AIMessage)
            and msg.content
            and len(msg.content) > 50
            and not msg.content.startswith("Transferring")
        ):
            return msg.content

    # Last resort.
    return messages[-1].content if messages else ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_swarm():
    """Create the multi-agent supervisor swarm.

    Builds three sub-agents (research, URL fetch, synthesizer), wires them
    under a ``langgraph-supervisor`` orchestrator, and compiles the graph
    into a runnable application.

    Returns:
        A compiled LangGraph ``CompiledStateGraph`` ready to be invoked
        with ``{"messages": [("user", "<question>")]}``.
    """
    model = ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        temperature=0,
    )

    # --- Sub-agents (use shared model, own tool instances) ----------------
    research_agent = create_react_agent(
        model=model,
        tools=[create_native_search_tool()],
        name="research_agent",
        prompt=RESEARCH_PROMPT,
    )

    url_agent = create_react_agent(
        model=model,
        tools=[create_url_fetch_tool()],
        name="url_agent",
        prompt=URL_PROMPT,
    )

    synthesizer_agent = create_synthesizer_agent()

    # --- Supervisor (auto-creates transfer_to_* handoff tools) ------------
    workflow = create_supervisor(
        agents=[research_agent, url_agent, synthesizer_agent],
        model=model,
        prompt=SUPERVISOR_PROMPT,
        output_mode="full_history",
    )

    app = workflow.compile()
    logger.info("swarm_created", model=settings.model_name, agent_count=3)
    return app


def run_swarm(question: str) -> str:
    """Run a question through the full multi-agent swarm.

    The supervisor routes the question to the appropriate agents and
    the synthesizer produces the final polished answer.

    Args:
        question: The user's natural-language question.

    Returns:
        The synthesizer's final answer as a string, including citations.
        On failure, returns a user-friendly error message.
    """
    logger.info("swarm_started", question=question)

    # Groq's Llama model occasionally generates malformed tool calls in the
    # supervisor context (400 tool_use_failed).  Retrying usually succeeds.
    max_attempts = 3
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            app = create_swarm()
            result = app.invoke({"messages": [("user", question)]})
            messages = result["messages"]

            # --- Optional RAG refinement pass -----------------------------
            if settings.rag_enabled:
                findings = _collect_agent_findings(messages)
                if findings:
                    rag_context = _rag_enhanced_synthesis(question, findings)
                    if rag_context:
                        logger.info("rag_synthesis_started")
                        synth = create_synthesizer_agent()
                        synth_result = synth.invoke(
                            {"messages": [
                                ("user", question),
                                ("user", rag_context),
                            ]},
                        )
                        answer = synth_result["messages"][-1].content
                        logger.info(
                            "swarm_finished",
                            answer_length=len(answer),
                            rag=True,
                        )
                        return answer

            # --- Fallback: extract from swarm messages (no RAG) -----------
            answer = _extract_answer(messages)

            logger.info("swarm_finished", answer_length=len(answer))
            return answer

        except Exception as exc:
            last_error = exc
            error_str = str(exc)

            if "429" in error_str or "rate" in error_str.lower():
                logger.warning("swarm_rate_limited", error=error_str)
                return (
                    "The LLM is temporarily rate-limited. "
                    "Please wait a moment and try again."
                )

            # Retry on Groq tool_use_failed (transient model formatting issue)
            if "tool_use_failed" in error_str and attempt < max_attempts:
                logger.warning(
                    "swarm_retry",
                    attempt=attempt,
                    error=type(exc).__name__,
                )
                continue

            logger.error(
                "swarm_failed",
                error=type(exc).__name__,
                detail=error_str,
            )
            return f"An error occurred while processing your question: {exc}"

    # All retries exhausted.
    logger.error("swarm_retries_exhausted", attempts=max_attempts)
    return f"An error occurred while processing your question: {last_error}"
