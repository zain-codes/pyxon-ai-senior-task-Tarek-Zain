"""Application settings powered by pydantic-settings.

Loads configuration from environment variables and an optional ``.env`` file.
All fields are validated on instantiation — missing required values
(``GROQ_API_KEY``, ``TAVILY_API_KEY``) raise a clear ``ValidationError``
at startup so problems surface immediately rather than at runtime.

Why a singleton + factory?
    ``get_settings()`` creates a *fresh* ``Settings`` instance every call,
    which is useful in tests that override env vars between assertions.
    The module-level ``settings`` object is a convenience singleton for
    production code that only needs one configuration.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Pyxon AI multi-agent system.

    Required environment variables (no defaults):
        GROQ_API_KEY   — API key for the Groq LLM provider.
        TAVILY_API_KEY — API key for the Tavily search service.

    All other variables have sensible defaults documented in ``.env.example``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM Provider (Groq) ------------------------------------------------
    groq_api_key: str = Field(
        ...,
        description="API key for the Groq LLM provider.",
    )
    model_name: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq-hosted model identifier.",
    )

    # --- Search Tool (Tavily) ------------------------------------------------
    tavily_api_key: str = Field(
        ...,
        description="API key for the Tavily search service.",
    )
    search_max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum search results per query.",
    )

    # --- URL Fetch Tool ------------------------------------------------------
    url_fetch_timeout: int = Field(
        default=10,
        ge=1,
        le=120,
        description="HTTP request timeout in seconds.",
    )
    url_fetch_max_size: int = Field(
        default=51200,
        ge=1024,
        le=10_485_760,
        description="Maximum response body size in bytes.",
    )

    # --- RAG / Embeddings ----------------------------------------------------
    rag_enabled: bool = Field(
        default=True,
        description="Enable the RAG vector-store pipeline.",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model for embeddings.",
    )

    # --- Logging -------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )


def get_settings() -> Settings:
    """Create a fresh ``Settings`` instance from the current environment.

    Useful in tests where environment variables are patched between calls.
    """
    return Settings()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
# In production the .env file (or real env vars) supplies the required keys.
# In test / CI environments the keys may be absent — the try/except lets
# the module import without crashing so that test code can patch settings
# before they are actually used.
# ---------------------------------------------------------------------------
try:
    settings = get_settings()
except Exception:
    settings = None  # type: ignore[assignment]
