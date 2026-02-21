"""Source credibility scoring based on domain tiers and recency."""
import re
from datetime import datetime, timedelta
from urllib.parse import urlparse

# Domain tier lists
TIER_1 = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nature.com", "sciencedirect.com", "who.int", "un.org",
    "nih.gov", "cdc.gov", "nasa.gov", "pubmed.ncbi.nlm.nih.gov",
    "nejm.org", "thelancet.com", "science.org",
}
TIER_2 = {
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    "economist.com", "ft.com", "bloomberg.com", "cnbc.com",
    "wikipedia.org", "en.wikipedia.org", "britannica.com",
    "pbs.org", "npr.org", "aljazeera.com",
    "thehindu.com", "ndtv.com", "livemint.com",
    "techcrunch.com", "arstechnica.com", "wired.com",
    "snopes.com", "factcheck.org", "politifact.com",
}
TIER_3 = {
    "medium.com", "substack.com", "wordpress.com",
    "forbes.com", "businessinsider.com", "huffpost.com",
    "indiatimes.com", "timesofindia.indiatimes.com",
}

# TLD bonuses
GOV_EDU_TLDS = {".gov", ".edu", ".ac.uk", ".gov.in", ".nic.in"}


def _get_domain(url):
    """Extract root domain from URL."""
    try:
        host = urlparse(url).hostname or ""
        host = host.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _domain_score(domain):
    """Score a domain based on tier membership."""
    if domain in TIER_1:
        return 10
    if domain in TIER_2:
        return 7
    if domain in TIER_3:
        return 4
    # Check TLD
    for tld in GOV_EDU_TLDS:
        if domain.endswith(tld):
            return 9
    # Check if subdomain of a tier-1/2 site
    for t1 in TIER_1:
        if domain.endswith("." + t1):
            return 9
    for t2 in TIER_2:
        if domain.endswith("." + t2):
            return 7
    return 2  # Unknown domain


def _recency_modifier(date_str):
    """Adjust score based on publication recency. Returns -1, 0, or +1."""
    if not date_str:
        return 0
    try:
        # Try common date formats
        for fmt in ("%Y-%m-%d", "%B %d, %Y", "%d %B %Y", "%Y-%m-%dT%H:%M:%S"):
            try:
                pub_date = datetime.strptime(date_str.strip()[:19], fmt)
                break
            except ValueError:
                continue
        else:
            return 0
        age = datetime.now() - pub_date
        if age < timedelta(days=180):
            return 1
        elif age > timedelta(days=730):
            return -1
        return 0
    except Exception:
        return 0


def score_source(url, date_str=""):
    """Score a source. Returns dict with score (0-10), tier, domain, details."""
    domain = _get_domain(url)
    base = _domain_score(domain)
    recency = _recency_modifier(date_str)
    final = max(0, min(10, base + recency))

    if base >= 9:
        tier = "Tier 1 - Highly Credible"
    elif base >= 6:
        tier = "Tier 2 - Credible"
    elif base >= 3:
        tier = "Tier 3 - Low Credibility"
    else:
        tier = "Tier 4 - Unreliable/Unknown"

    return {
        "domain": domain,
        "score": final,
        "tier": tier,
        "recency_modifier": recency,
    }
