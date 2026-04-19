import google.generativeai as genai
import os

def refine_decision(prediction, confidence, evidence):
    """
    Milestone 2 Agentic Node: 
    Uses Gemini AI to autonomously reason about content veracity.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Context for the agent
            source_context = "\n".join([f"- Title: {e['title']} | Snippet: {e['snippet']}" for e in evidence])
            
            prompt = f"""
            SYSTEM ROLE: Sovereign Fact Verification Engine
            
            ML SIGNAL: {prediction} ({confidence}% confidence)
            LIVE WEB EVIDENCE:
            {source_context if evidence else "No direct evidence found."}
            
            TASK: 
            1. Verify the factual accuracy of the claim.
            2. If there is a hallucination or error (e.g. wrong PM, wrong country), EXPLICITLY state the correct answer.
            3. Provide a clear verdict: [VERIFIED], [FAKE], or [MISLEADING].
            4. Keep the reasoning simple and professional.
            """
            
            response = model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            return f"Agent Error: {str(e)}"

    # Fallback if no API key exists
    if prediction == "Real" and len(evidence) > 0:
        return "[Likely Real] Content aligns with foundational web evidence."
    
    return "[Investigation Req] Patterns of misinformation detected or lack of evidence."