from fastapi import APIRouter
from app.agent.agent_loop import run_agent

router = APIRouter()

@router.post("/analyze")
async def analyze_news(data: dict):
    input_text = data.get("text")
    url = data.get("url")

    result = await run_agent(input_text, url)
    return result