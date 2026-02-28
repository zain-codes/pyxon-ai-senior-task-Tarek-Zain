"""Shared test fixtures — ensures tests run without real API keys."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _fake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject dummy API keys so pydantic-settings validation passes.

    ``autouse=True`` means every test in the suite gets these env vars
    automatically — no test can accidentally hit a real API.
    """
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
