def format_output(text, prediction, confidence, evidence, final_decision):
    """
    Standardizes the output into two distinct modules:
    1. News Verification (ML/Linguistic)
    2. Fact Verification (Search/Agentic)
    """
    # Fix for float/string confidence — convert 0-1 range to percentage
    try:
        if isinstance(confidence, str):
            conf_val = float(confidence.replace('%', ''))
        else:
            conf_val = float(confidence)
        
        # If the value is between 0 and 1, convert to percentage
        if 0 < conf_val <= 1:
            conf_val = conf_val * 100
    except:
        conf_val = 0.0

    return {
        "text": text,
        
        # Module 1: News Verification (Milestone 1)
        "news_verification": {
            "prediction": prediction,
            "confidence": f"{conf_val:.1f}%",
            "assessment": "Linguistic pattern analysis complete."
        },
        
        # Module 2: Fact Verification (Milestone 2)
        "fact_verification": {
            "evidence": evidence,
            "agent_verdict": final_decision,
            "sources_found": len(evidence)
        },
        
        # Legacy compatibility
        "prediction": prediction,
        "confidence": conf_val,
        "decision": final_decision,
        "evidence": evidence
    }