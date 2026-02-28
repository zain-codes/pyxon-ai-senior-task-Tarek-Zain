"""Structured logging configuration using structlog.

Provides human-readable, colourful console output in development.
Can be extended to JSON output for production log aggregation.

Usage::

    from src.config.logging import setup_logging, get_logger

    setup_logging("DEBUG")
    logger = get_logger(__name__)
    logger.info("server started", port=8000)
"""

from __future__ import annotations

import logging

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog processors and minimum log level.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structlog logger.

    Args:
        name: Logger name — typically ``__name__`` of the calling module.
    """
    return structlog.get_logger(name)
