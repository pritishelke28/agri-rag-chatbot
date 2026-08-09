"""
FastAPI backend for the Agricultural Advisory Chatbot.

Serves the static frontend (frontend/index.html) and exposes:
  - GET  /api/health         readiness check for each subsystem
  - POST /api/advisory       text-based requests: general / crop / fertilizer / disease-by-symptom
  - POST /api/diagnose-image multipart request: disease diagnosis from a leaf photo

Run:
    python src/api.py
Then open:
    http://localhost:8000
"""
import sys
import tempfile
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Agricultural Advisory Chatbot API")

FRONTEND_DIR = config.ROOT_DIR / "frontend"

# ---------------------------------------------------------------------------
# Lazy singletons — heavy objects (embeddings model, vector store, LLM
# client) are built once on first use and reused, instead of reloading on
# every request.
# ---------------------------------------------------------------------------
_rag_chain = None


def get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        if not config.GROQ_API_KEY:
            raise HTTPException(
                status_code=503,
                detail="GROQ_API_KEY is not set. Add it to your .env file and restart the server.",
            )
        from src.rag.chain import build_rag_chain
        try:
            _rag_chain = build_rag_chain()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"RAG chain unavailable — has ingestion been run? ({e})",
            )
    return _rag_chain


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class AdvisoryRequest(BaseModel):
    module: str  # "general" | "crop" | "fertilizer" | "disease"
    question: Optional[str] = ""
    fields: Optional[dict] = None


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(index_path)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    vectorstore_ready = config.CHROMA_PERSIST_DIR.exists() and any(config.CHROMA_PERSIST_DIR.iterdir())
    crop_model_ready = (config.MODELS_DIR / "crop_recommender.pkl").exists()
    disease_model_ready = (config.MODELS_DIR / "disease_classifier.pt").exists()
    return {
        "groq_key_set": bool(config.GROQ_API_KEY),
        "vectorstore_ready": vectorstore_ready,
        "crop_model_ready": crop_model_ready,
        "disease_model_ready": disease_model_ready,
    }


# ---------------------------------------------------------------------------
# Main advisory endpoint (text-based modules)
# ---------------------------------------------------------------------------
@app.post("/api/advisory")
async def advisory(payload: AdvisoryRequest):
    module = payload.module
    fields = payload.fields or {}

    if module == "general":
        chain = get_rag_chain()
        if not payload.question:
            raise HTTPException(status_code=400, detail="question is required for the general module")
        answer = chain.invoke(payload.question)
        return {"text": answer}

    elif module == "crop":
        from src.models.crop_recommender import predict_top_k
        required = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        missing = [f for f in required if f not in fields]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")
        try:
            results = predict_top_k(fields, k=3)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))

        text = "Based on the soil values entered, the top suitable crops are:"
        return {
            "text": text,
            "top_crops": [{"crop": c, "probability": round(p, 3)} for c, p in results],
            "nutrients": {"N": fields["N"], "P": fields["P"], "K": fields["K"]},
        }

    elif module == "fertilizer":
        from src.models.fertilizer_advisor import recommend_fertilizer_dosage
        crop = fields.get("crop")
        if not crop:
            raise HTTPException(status_code=400, detail="fields.crop is required")
        current = {
            "N": fields.get("N", 0),
            "P": fields.get("P", 0),
            "K": fields.get("K", 0),
        }
        try:
            plan = recommend_fertilizer_dosage(current, crop)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=503, detail=str(e))

        return {
            "text": f"Recommended dosage plan for {crop}:",
            "fertilizer_plan": plan["fertilizer_plan_kg_per_acre"],
            "nutrients": current,
        }

    elif module == "disease":
        if not payload.question:
            raise HTTPException(status_code=400, detail="Describe symptoms in 'question', or use /api/diagnose-image for a photo")
        from src.models.disease_classifier import predict_from_symptoms
        chain = get_rag_chain()
        answer = predict_from_symptoms(payload.question, chain=chain)
        return {"text": answer}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown module '{module}'")


# ---------------------------------------------------------------------------
# Disease diagnosis from an uploaded leaf photo
# ---------------------------------------------------------------------------
@app.post("/api/diagnose-image")
async def diagnose_image(file: UploadFile = File(...)):
    from src.models.disease_classifier import predict

    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        results = predict(tmp_path, top_k=3)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Disease image model isn't trained yet ({e}). "
                   f"Try describing the symptoms as text instead.",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "text": "Diagnosis based on the uploaded leaf photo:",
        "diagnosis": [{"label": label, "probability": round(p, 3)} for label, p in results],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)