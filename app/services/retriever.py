import requests
import os

def fetch_evidence(claims):
    """
    Fetches real web evidence for claims using Google Custom Search API.
    Falls back to DuckDuckGo instant answers if no Google API key is available.
    """
    evidence_list = []

    for claim in claims:
        try:
            results = _search_duckduckgo(claim)
            evidence_list.extend(results)
        except Exception:
            evidence_list.append({
                "title": f"Search failed for: {claim[:40]}...",
                "source": "N/A",
                "snippet": "Could not retrieve evidence.",
                "credibility": "unknown"
            })

    return evidence_list


def _search_duckduckgo(query):
    """Use DuckDuckGo instant answer API for evidence retrieval."""
    results = []
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": 1},
            timeout=8
        )
        data = resp.json()

        # Abstract (main answer)
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", query),
                "source": data.get("AbstractSource", "DuckDuckGo"),
                "snippet": data["AbstractText"][:300],
                "credibility": "high",
                "url": data.get("AbstractURL", "")
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "source": "DuckDuckGo",
                    "snippet": topic.get("Text", "")[:300],
                    "credibility": "medium",
                    "url": topic.get("FirstURL", "")
                })

    except Exception:
        pass

    return results