from fastapi import APIRouter, File, UploadFile, HTTPException
from app.agent.agent_loop import run_agent
import io

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

router = APIRouter()

@router.post("/analyze")
async def analyze_news(data: dict):
    input_text = data.get("text")
    url = data.get("url")
    mode = data.get("mode", "both")

    result = await run_agent(input_text, url, mode=mode)
    return result

@router.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    if PdfReader is None:
        raise HTTPException(status_code=500, detail="pypdf library is not installed.")

    try:
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
        
        result = await run_agent(text, None)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")