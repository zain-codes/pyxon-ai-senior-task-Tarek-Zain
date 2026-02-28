"""Tests for tool wrappers (search_tool, url_fetch_tool).

All external calls (Tavily API, httpx requests) are mocked so these
tests run without real API keys.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Search tool tests
# ---------------------------------------------------------------------------


class TestSearchTool:
    """Tests for ``create_search_tool()``."""

    @patch("src.tools.search_tool.TavilySearch")
    def test_returns_results(self, mock_tavily_cls: MagicMock) -> None:
        """Search tool returns Tavily results for a normal query."""
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = [
            {"title": "AI News", "url": "https://example.com", "content": "AI info"},
        ]
        mock_tavily_cls.return_value = mock_instance

        from src.tools.search_tool import create_search_tool

        tool = create_search_tool()
        result = tool.invoke({"query": "test query"})

        assert result is not None
        mock_instance.invoke.assert_called_once()

    @patch("src.tools.search_tool.TavilySearch")
    def test_handles_error(self, mock_tavily_cls: MagicMock) -> None:
        """Search tool returns a friendly error string on API failure."""
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = Exception("API error")
        mock_tavily_cls.return_value = mock_instance

        from src.tools.search_tool import create_search_tool

        tool = create_search_tool()
        result = tool.invoke({"query": "failing query"})

        assert "Search failed" in result
        assert "API error" in result

    @patch("src.tools.search_tool.TavilySearch")
    def test_empty_results(self, mock_tavily_cls: MagicMock) -> None:
        """Search tool handles empty result list gracefully."""
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = []
        mock_tavily_cls.return_value = mock_instance

        from src.tools.search_tool import create_search_tool

        tool = create_search_tool()
        result = tool.invoke({"query": "obscure query"})

        assert result is not None  # Should not crash


# ---------------------------------------------------------------------------
# URL fetch tool tests
# ---------------------------------------------------------------------------


def _make_mock_response(
    *,
    content: bytes,
    content_type: str = "text/html",
    status_code: int = 200,
) -> MagicMock:
    """Build a mock that mimics ``httpx.Client().stream()``'s context-manager chain."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"content-type": content_type}
    response.raise_for_status = MagicMock()
    response.iter_bytes.return_value = iter([content])
    return response


class TestURLTool:
    """Tests for ``create_url_fetch_tool()``."""

    @patch("src.tools.url_fetch_tool.httpx.Client")
    def test_json_response(self, mock_client_cls: MagicMock) -> None:
        """URL tool parses and pretty-prints JSON API responses."""
        payload = json.dumps({"id": 1, "name": "test"}).encode()
        mock_resp = _make_mock_response(
            content=payload, content_type="application/json"
        )

        # Wire up the double context-manager: Client().__enter__().stream().__enter__()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.stream = MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=mock_resp),
            __exit__=MagicMock(return_value=False),
        ))
        mock_client_cls.return_value = mock_client

        from src.tools.url_fetch_tool import create_url_fetch_tool

        tool = create_url_fetch_tool()
        result = tool.invoke({"url": "https://api.example.com/data"})

        assert '"id": 1' in result
        assert '"name": "test"' in result

    @patch("src.tools.url_fetch_tool.httpx.Client")
    def test_html_response(self, mock_client_cls: MagicMock) -> None:
        """URL tool extracts text from HTML pages."""
        html = b"<html><body><p>Hello world from the web page</p></body></html>"
        mock_resp = _make_mock_response(content=html, content_type="text/html")

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.stream = MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=mock_resp),
            __exit__=MagicMock(return_value=False),
        ))
        mock_client_cls.return_value = mock_client

        from src.tools.url_fetch_tool import create_url_fetch_tool

        tool = create_url_fetch_tool()
        result = tool.invoke({"url": "https://example.com/page"})

        assert "Hello world" in result

    @patch("src.tools.url_fetch_tool.httpx.Client")
    def test_timeout(self, mock_client_cls: MagicMock) -> None:
        """URL tool returns a friendly message on timeout."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.stream = MagicMock(
            side_effect=httpx.TimeoutException("timed out")
        )
        mock_client_cls.return_value = mock_client

        from src.tools.url_fetch_tool import create_url_fetch_tool

        tool = create_url_fetch_tool()
        result = tool.invoke({"url": "https://slow.example.com"})

        assert "timed out" in result.lower()

    @patch("src.tools.url_fetch_tool.httpx.Client")
    def test_large_response_truncated(self, mock_client_cls: MagicMock) -> None:
        """URL tool truncates responses exceeding _MAX_CONTENT_CHARS."""
        large_body = ("x" * 5000).encode()
        mock_resp = _make_mock_response(content=large_body, content_type="text/plain")

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.stream = MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=mock_resp),
            __exit__=MagicMock(return_value=False),
        ))
        mock_client_cls.return_value = mock_client

        from src.tools.url_fetch_tool import create_url_fetch_tool

        tool = create_url_fetch_tool()
        result = tool.invoke({"url": "https://example.com/huge"})

        assert "truncated" in result.lower()
