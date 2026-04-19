---
title: TruthGuard AI Fake News Detection
emoji: 🕵️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---
# Fake News Detection Agent

AI-powered agent that combines a **fine-tuned DistilBert** classifier with web retrieval and multi-step reasoning to detect fake news.

## Architecture

```
app/
├── main.py                 # FastAPI entry point
├── routes/
│   └── analyze.py          # API endpoints
├── agent/
│   ├── agent_loop.py       # Core reasoning loop
│   └── decision.py         # Decision logic
├── models/
│   └── distilbert.py       # DistilBert model wrapper
├── services/
│   ├── retriever.py        # Fetch external evidence
│   ├── scraper.py          # URL → text extraction
│   └── nlp_utils.py        # Claim extraction & NLP
└── utils/
    └── formatter.py        # Structured output formatting
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.  
Interactive docs at `http://127.0.0.1:8000/docs`.

## API Endpoints

| Method | Path             | Description                              |
|--------|------------------|------------------------------------------|
| POST   | `/api/analyze`   | Full agent analysis (model + web search) |
| POST   | `/api/predict`   | Quick model-only prediction              |
| GET    | `/api/stats`     | Model info + session analytics           |

### Example — Full Analysis

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking: Scientists discover new planet in habitable zone"}'
```

### Example — Quick Predict

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Government secretly controls weather patterns"}'
```

### Example — Analyze from URL

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

## Agent Pipeline

1. **Input Processing** — accepts raw text or scrapes a URL
2. **Claim Extraction** — splits text into factual claims
3. **DistilBert Prediction** — classifies with attention-based explanations
4. **Web Retrieval** — searches for corroborating evidence from trusted sources
5. **Decision Synthesis** — combines model + evidence into a reasoned verdict
