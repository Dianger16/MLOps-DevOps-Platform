# api/serve.py
# FastAPI app that serves the trained sentiment model
# Endpoints: POST /predict, GET /health, GET /metrics, GET /model-info

import os
import time
import pickle
import logging
from datetime import datetime, timezone
from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="Sentiment Analysis API",
    description="MLOps Platform — Sentiment Analysis Model Serving",
    version=os.getenv("APP_VERSION", "1.0.0"),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model + metrics ────────────────────────────────────────────────────
model = None
request_count    = 0
prediction_times = []
label_counts     = defaultdict(int)
start_time       = time.time()


def load_model():
    """Load model from local pickle file"""
    global model
    model_path = os.getenv("MODEL_PATH", "model/artifacts/model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run: python model/train.py"
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from: {model_path}")





# ── Request / Response models ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, json_schema_extra={"example": "This product is amazing!"})

class PredictResponse(BaseModel):
    text:        str
    sentiment:   str
    confidence:  float
    probabilities: dict
    serving_model_version: str
    latency_ms:  float
    timestamp:   str

class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":        "healthy",
        "model_loaded":  model is not None,
        "uptime_seconds": round(time.time() - start_time),
        "version":       os.getenv("APP_VERSION", "1.0.0"),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    global request_count
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.time()
    try:
        proba  = model.predict_proba([request.text])[0]
        labels = model.classes_
        label_proba = dict(zip(labels, [round(float(p), 4) for p in proba]))
        sentiment   = max(label_proba, key=label_proba.get)
        confidence  = label_proba[sentiment]

        latency_ms = round((time.time() - t0) * 1000, 2)

        # Track metrics
        request_count += 1
        prediction_times.append(latency_ms)
        label_counts[sentiment] += 1

        return PredictResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence,
            probabilities=label_proba,
            serving_model_version=os.getenv("MODEL_VERSION", "1.0.0"),
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for text in request.texts:
        proba       = model.predict_proba([text])[0]
        labels      = model.classes_
        label_proba = dict(zip(labels, [round(float(p), 4) for p in proba]))
        sentiment   = max(label_proba, key=label_proba.get)
        results.append({
            "text":          text,
            "sentiment":     sentiment,
            "confidence":    label_proba[sentiment],
            "probabilities": label_proba,
        })
    return {"predictions": results, "count": len(results)}


@app.get("/metrics")
def metrics():
    """Prometheus-style metrics for monitoring"""
    avg_latency = (
        round(sum(prediction_times) / len(prediction_times), 2)
        if prediction_times else 0
    )
    return {
        "total_requests":    request_count,
        "avg_latency_ms":    avg_latency,
        "uptime_seconds":    round(time.time() - start_time),
        "predictions_by_label": dict(label_counts),
        "serving_model_version":     os.getenv("MODEL_VERSION", "1.0.0"),
    }


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type":     type(model).__name__,
        "pipeline_steps": [step[0] for step in model.steps],
        "classes":        list(model.classes_),
        "version":        os.getenv("MODEL_VERSION", "1.0.0"),
        "mlflow_uri":     os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    }