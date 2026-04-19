from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv
from app.routes.analyze import router

# Load API Keys from .env
load_dotenv()

app = FastAPI(title="Intelligent News Monitor")

# Mount frontend directory for static assets
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))