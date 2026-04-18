def format_output(text, prediction, confidence, evidence, decision):
    return {
        "prediction": prediction,
        "confidence": f"{round(confidence * 100, 2)}%",
        "decision": decision,
        "evidence": evidence[:5]
    }