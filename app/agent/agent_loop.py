from app.services.scraper import scrape_url
from app.services.nlp_utils import extract_claims
from app.models.distilbert import predict
from app.services.retriever import fetch_evidence
from app.agent.decision import refine_decision
from app.utils.formatter import format_output

CONF_THRESHOLD = 0.85

async def run_agent(text=None, url=None):

    if url:
        result = scrape_url(url)
        if result.get("success"):
            text = result.get("text")
    
    if not text:
        return {"error": "No text or URL provided"}

    claims = extract_claims(text)

    # Step 3: ML Prediction
    prediction, confidence = predict(text)

    # Step 4: Retrieve evidence
    evidence = fetch_evidence(claims)

    # Step 5: Agent reasoning loop
    final_decision = refine_decision(
        prediction,
        confidence,
        evidence
    )

    # Step 6: Structured output
    return format_output(
        text,
        prediction,
        confidence,
        evidence,
        final_decision
    )