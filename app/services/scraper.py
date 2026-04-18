"""
URL scraper — fetches a web page and extracts its visible text content.
"""

import re
import requests
from html.parser import HTMLParser


# Tags whose content we skip entirely
_SKIP_TAGS = {
    "script", "style", "noscript", "header", "footer", "nav",
    "aside", "iframe", "svg", "form",
}


class _TextExtractor(HTMLParser):
    """Lightweight HTML→text parser (no external deps like BeautifulSoup)."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        return " ".join(self._parts)


def scrape_url(url: str, timeout: int = 10) -> dict:
    """
    Fetch a URL and extract its main text content.

    Returns:
        dict with keys: url, title, text, word_count, success, error
    """
    result = {
        "url": url,
        "title": "",
        "text": "",
        "word_count": 0,
        "success": False,
        "error": None,
    }

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()

        html = resp.text

        # Extract <title>
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
        if title_match:
            result["title"] = title_match.group(1).strip()

        # Extract body text
        parser = _TextExtractor()
        parser.feed(html)
        text = parser.get_text()

        # Basic cleanup — collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        result["text"] = text
        result["word_count"] = len(text.split())
        result["success"] = True

    except requests.RequestException as exc:
        result["error"] = str(exc)

    return result
