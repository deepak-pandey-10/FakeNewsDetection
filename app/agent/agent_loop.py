from app.services.scraper import scrape_url
from app.services.nlp_utils import extract_claims
from app.services.retriever import fetch_evidence
from app.agent.decision import refine_decision
from app.utils.formatter import format_output

async def run_agent(text=None, url=None, mode="both"):
    """
    Orchestrator - Now entirely API-driven.
    """

    if url:
        result = scrape_url(url)
        if result.get("success"):
            text = result.get("text")
    
    if not text:
        return {"error": "No text or URL provided"}

    # Initial defaults
    prediction = "N/A"
    confidence = 0.0
    evidence = []
    
    # Part 1: News Verification (Model) - REMOVED
    # We ignore the 'mode' parameter now and always run fact checking.

    # Part 2: Fact Verification (Agent/Search)
    claims = extract_claims(text)
    evidence = fetch_evidence(claims)
    
    final_decision = refine_decision(
        prediction,
        confidence,
        evidence,
        original_text=text
    )

    # Step 6: Structured output
    return format_output(
        text,
        prediction,
        confidence,
        evidence,
        final_decision
    )