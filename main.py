"""
Pyxon AI Senior Task — End-to-End Demo
=======================================
This script demonstrates all three agents and the multi-agent swarm.

Usage:
    python main.py                    # Run all demo scenarios
    python main.py --interactive      # Interactive mode (ask your own questions)
"""

from __future__ import annotations

import argparse
import sys


def run_demo() -> None:
    """Run 3 pre-configured demo scenarios showing each capability."""
    from src.agents.search_agent import run_search_agent
    from src.agents.url_agent import run_url_agent
    from src.swarm.supervisor import run_swarm

    print("=" * 60)
    print("  Pyxon AI — Multi-Agent System Demo")
    print("=" * 60)

    # Demo 1: Search Agent (standalone)
    print("\n--- DEMO 1: Search Agent (Tavily + Groq) ---")
    question1 = "What are the latest developments in AI regulation?"
    print(f"Question: {question1}\n")
    answer1 = run_search_agent(question1)
    print(f"Answer:\n{answer1}\n")

    # Demo 2: URL Agent (standalone)
    print("\n--- DEMO 2: URL Agent (httpx + trafilatura) ---")
    question2 = (
        "What does https://jsonplaceholder.typicode.com/posts/1 return?"
    )
    print(f"Question: {question2}\n")
    answer2 = run_url_agent(question2)
    print(f"Answer:\n{answer2}\n")

    # Demo 3: Multi-Agent Swarm (supervisor pattern)
    # Uses a search-focused question for reliable demo output.
    # The swarm also supports URL fetching — try it in interactive mode
    # with questions like "Fetch https://api.github.com and describe it".
    print("\n--- DEMO 3: Multi-Agent Swarm (supervisor + RAG) ---")
    question3 = (
        "What is LangGraph, what problem does it solve, "
        "and how does it compare to other agent frameworks?"
    )
    print(f"Question: {question3}\n")
    answer3 = run_swarm(question3)
    print(f"Answer:\n{answer3}\n")

    print("=" * 60)
    print("  All demos completed successfully!")
    print("=" * 60)


def run_interactive() -> None:
    """Interactive mode — user types questions, swarm answers."""
    from src.swarm.supervisor import run_swarm

    print("=" * 60)
    print("  Pyxon AI — Interactive Mode")
    print("  Type a question and press Enter. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            question = input("\nYour question: ").strip()
        except EOFError:
            break

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        print("Processing with multi-agent swarm...\n")
        answer = run_swarm(question)
        print(f"Answer:\n{answer}")


def main() -> None:
    """Entry point — parse arguments, initialise logging, run demo."""
    parser = argparse.ArgumentParser(
        description="Pyxon AI Multi-Agent Demo",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode",
    )
    args = parser.parse_args()

    # Settings and logging are imported inside try/except so a missing
    # .env file produces a helpful message instead of a traceback.
    try:
        from src.config.settings import settings

        if settings is None:
            raise RuntimeError(
                "Settings could not be loaded. "
                "Make sure your .env file has GROQ_API_KEY and TAVILY_API_KEY."
            )

        from src.config.logging import setup_logging, get_logger

        setup_logging(settings.log_level)
        logger = get_logger(__name__)
    except Exception as exc:
        print(f"\nConfiguration error: {exc}")
        print(
            "\nQuick fix:\n"
            "  1. cp .env.example .env\n"
            "  2. Edit .env and add your GROQ_API_KEY and TAVILY_API_KEY\n"
            "     - Groq key: https://console.groq.com/keys\n"
            "     - Tavily key: https://app.tavily.com\n"
        )
        sys.exit(1)

    logger.info("starting_demo", mode="interactive" if args.interactive else "demo")

    try:
        if args.interactive:
            run_interactive()
        else:
            run_demo()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as exc:
        logger.error("demo_failed", error=str(exc))
        print(f"\nError: {exc}")
        print("Make sure your .env file has valid GROQ_API_KEY and TAVILY_API_KEY.")
        sys.exit(1)


if __name__ == "__main__":
    main()
