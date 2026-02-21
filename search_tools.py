"""DuckDuckGo search and URL content fetching."""
import re
import urllib.request
import urllib.error
from html.parser import HTMLParser

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
from config import SEARCH_RESULTS_PER_QUERY


class _TextExtractor(HTMLParser):
    """Simple HTML to text converter."""
    def __init__(self):
        super().__init__()
        self._text = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._text.append(data)

    def get_text(self):
        return " ".join(self._text)


def search_ddg(query, max_results=SEARCH_RESULTS_PER_QUERY):
    """Search DuckDuckGo and return list of {title, url, snippet}."""
    try:
        results = list(DDGS().text(query, max_results=max_results))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except Exception as e:
        return [{"title": "Search error", "url": "", "snippet": str(e)}]


def fetch_url_text(url, max_chars=3000):
    """Fetch a URL and return extracted text (truncated)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        parser = _TextExtractor()
        parser.feed(html)
        text = parser.get_text()
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""
