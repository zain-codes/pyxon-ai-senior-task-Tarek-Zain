"""URL fetch tool — retrieves and extracts content from web pages and APIs.

Why httpx over requests:
    httpx supports async I/O, has built-in streaming (to enforce size limits
    without downloading the entire response), and provides first-class timeout
    configuration per-request.

Why trafilatura over raw HTML:
    Raw HTML is cluttered with navigation, ads, scripts, and footers.
    trafilatura uses heuristics and readability algorithms to extract just
    the article / main content — exactly what an LLM needs to reason about.
    BeautifulSoup is kept as a fallback for pages trafilatura cannot parse.

Security considerations:
    Fetching arbitrary URLs on behalf of an LLM carries risk (SSRF, large
    responses, slow servers).  Mitigations applied here:
    - Hard timeout from ``settings.url_fetch_timeout`` (default 10 s).
    - Streaming read capped at ``settings.url_fetch_max_size`` (default 50 KB)
      to prevent memory exhaustion from unbounded downloads.
    - Descriptive User-Agent header so site operators can identify the bot.
    - All exceptions are caught and returned as strings — the tool never raises.
"""

from __future__ import annotations

import json

import httpx
import trafilatura
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool, tool

from src.config.logging import get_logger
from src.config.settings import settings

logger = get_logger(__name__)

_MAX_CONTENT_CHARS = 4000


def create_url_fetch_tool() -> BaseTool:
    """Create a URL-fetching tool configured from application settings.

    Returns:
        A LangChain ``BaseTool`` that accepts a URL string and returns
        the extracted content or an error message.
    """

    @tool("fetch_url")
    def fetch_url(url: str) -> str:
        """Fetch a URL (web page or API endpoint) and return its content.

        Handles JSON APIs (pretty-printed) and HTML pages (article text
        extracted via trafilatura). Responses are truncated to 4 000
        characters to avoid flooding the LLM context window.

        Args:
            url: The fully-qualified URL to fetch.

        Returns:
            Extracted content string, or an error description if the
            request fails.
        """
        logger.info("url_fetch_started", url=url)

        try:
            with httpx.Client(
                timeout=settings.url_fetch_timeout,
                follow_redirects=True,
            ) as client:
                with client.stream(
                    "GET",
                    url,
                    headers={"User-Agent": "PyxonAI-Agent/1.0"},
                ) as response:
                    response.raise_for_status()

                    # Read up to max_size bytes to guard against huge responses.
                    chunks: list[bytes] = []
                    bytes_read = 0
                    for chunk in response.iter_bytes(chunk_size=8192):
                        chunks.append(chunk)
                        bytes_read += len(chunk)
                        if bytes_read >= settings.url_fetch_max_size:
                            break

                    raw_bytes = b"".join(chunks)

            content_type = response.headers.get("content-type", "")
            status_code = response.status_code

            logger.info(
                "url_fetch_response",
                url=url,
                status_code=status_code,
                content_type=content_type,
                content_length=len(raw_bytes),
            )

        except httpx.TimeoutException:
            logger.error("url_fetch_timeout", url=url)
            return (
                f"Error fetching URL: Connection timed out after "
                f"{settings.url_fetch_timeout} seconds."
            )
        except httpx.HTTPStatusError as exc:
            logger.error(
                "url_fetch_http_error",
                url=url,
                status_code=exc.response.status_code,
            )
            return (
                f"Error fetching URL: HTTP {exc.response.status_code} "
                f"({exc.response.reason_phrase})."
            )
        except httpx.ConnectError:
            logger.error("url_fetch_connect_error", url=url)
            return (
                "Error fetching URL: Could not connect to the server. "
                "The domain may not exist or the server is down."
            )
        except Exception as exc:
            logger.error("url_fetch_failed", url=url, error=str(exc))
            return f"Error fetching URL: {exc}"

        # --- Extract content based on Content-Type -----------------------
        text = raw_bytes.decode("utf-8", errors="replace")

        if "json" in content_type:
            content = _extract_json(text)
        else:
            content = _extract_html(text)

        return _truncate(content)

    return fetch_url


def _extract_json(text: str) -> str:
    """Parse and pretty-print a JSON response."""
    try:
        data = json.loads(text)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return text


def _extract_html(html: str) -> str:
    """Extract readable text from HTML using trafilatura, with BS4 fallback."""
    extracted = trafilatura.extract(html)
    if extracted:
        return extracted

    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _truncate(content: str) -> str:
    """Truncate content to ``_MAX_CONTENT_CHARS`` if needed."""
    if len(content) <= _MAX_CONTENT_CHARS:
        return content
    return content[:_MAX_CONTENT_CHARS] + (
        "\n... [truncated — response too large, "
        f"showing first {_MAX_CONTENT_CHARS} chars]"
    )
