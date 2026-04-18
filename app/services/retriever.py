import requests

def fetch_evidence(claims):
    evidence_list = []

    for claim in claims:
        # Mocking evidence retrieval for the sake of the demo
        evidence_list.append({
            "title": f"Report on: {claim[:30]}...",
            "source": "BBC News",
            "credibility": "high"
        })
        evidence_list.append({
            "title": "Unverified blog post about the issue",
            "source": "Random Blog",
            "credibility": "low"
        })

    return evidence_list