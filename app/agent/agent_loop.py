from app.services.scraper import scrape_url
from app.services.nlp_utils import extract_claims
from app.models.distilbert import predict
from app.services.retriever import fetch_evidence
from app.agent.decision import refine_decision
from app.utils.formatter import format_output

async def run_agent(text=None, url=None, mode="both"):
    """
    Milestone 1/2 Orchestrator
    Explicit Mode Control:
    - 'news': Only runs ML prediction (Milestone 1)
    - 'fact': Only runs web search & reasoning (Milestone 2)
    - 'both': Runs the full pipeline
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
    final_decision = "Analysis not requested."

    # Part 1: News Verification (Model)
    if mode in ["news", "both"]:
        prediction, confidence = predict(text)
        if mode == "news":
            final_decision = "Linguistic analysis complete. Factual retrieval skipped."

    # Part 2: Fact Verification (Agent/Search)
    if mode in ["fact", "both"]:
        claims = extract_claims(text)
        evidence = fetch_evidence(claims)
        
        # If we already have a prediction from Part 1, use it. 
        # Otherwise, assume N/A for style.
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