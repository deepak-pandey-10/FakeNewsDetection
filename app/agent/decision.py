def refine_decision(prediction, confidence, evidence):
    
    if confidence < 0.6:
        return "Uncertain"

    if not evidence:
        return "Suspicious"

    trusted_sources = [
        e for e in evidence if e["credibility"] == "high"
    ]

    if prediction == "Fake" and len(trusted_sources) == 0:
        return "Likely Fake"

    if prediction == "Real" and len(trusted_sources) > 0:
        return "Likely Real"

    return "Needs Further Verification"