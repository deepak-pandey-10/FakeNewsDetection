from google import genai
import os
import time

# Models in order of preference — gemma-3-27b-it works when gemini quota is exhausted
MODELS_TO_TRY = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemma-3-27b-it"]

def refine_decision(prediction, confidence, evidence, original_text=""):
    """
    Uses Gemini/Gemma AI to verify factual accuracy of claims.
    Falls back through multiple models if quota is exhausted.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return _no_api_fallback(prediction, confidence)

    last_error = ""
    
    for attempt in range(2):
        for model_name in MODELS_TO_TRY:
            try:
                client = genai.Client(api_key=api_key)
                
                # Build evidence context
                if evidence:
                    source_context = "\n".join([
                        f"- Source: {e.get('source', 'N/A')} | Info: {e.get('snippet', e.get('title', ''))}"
                        for e in evidence
                    ])
                else:
                    source_context = "No direct web evidence found."
                
                prompt = f"""SYSTEM ROLE: You are a professional Fact Verification Engine.

CLAIM TO VERIFY:
"{original_text}"

WEB EVIDENCE GATHERED:
{source_context}

YOUR TASK:
1. Analyze the claim above for FACTUAL ACCURACY using your knowledge.
2. If the claim contains factual errors (wrong names, wrong dates, wrong facts, wrong capitals, wrong leaders, etc.), EXPLICITLY state what is wrong and provide the correct information.
3. If the claim is accurate, confirm it with supporting reasoning.
4. Provide a clear verdict at the START: [VERIFIED], [FAKE], or [MISLEADING].
5. Keep your response concise (3-5 sentences max).
6. Be specific — always cite the correct facts when correcting errors.

RESPOND NOW:"""
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text.strip()
                
            except Exception as e:
                last_error = str(e)
                if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
                    continue  # Try next model
                elif "404" in last_error:
                    continue  # Model not available
                elif "400" in last_error and "expired" in last_error.lower():
                    break  # Key expired
                return f"Agent Error: {last_error}"
        
        # All models failed this attempt, brief wait
        if attempt < 1:
            time.sleep(3)

    return f"[UNVERIFIED] All AI models are currently unavailable. Last error: {last_error[:80]}"


def _no_api_fallback(prediction, confidence):
    """Fallback when no API key is configured."""
    return (
        "[UNVERIFIED] No Gemini API key configured. "
        "Please add GEMINI_API_KEY to your .env file to enable AI fact-checking."
    )