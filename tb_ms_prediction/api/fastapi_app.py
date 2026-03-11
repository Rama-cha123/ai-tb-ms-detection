"""
fastapi_app.py
==============
REST API for TB and MS predictions using FastAPI.

Launch:
    uvicorn api.fastapi_app:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /              → API info
    GET  /health        → Health check
    POST /predict       → Predict from uploaded image file
    POST /predict/base64 → Predict from base64-encoded image
"""

import os
import sys
import io
import base64
import numpy as np
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DISEASE_CONFIG, PREDICTION_THRESHOLD

# ── App setup ────────────────────────────────────────────────
app = FastAPI(
    title="TB & MS Medical AI Screening API",
    description="REST API for Tuberculosis and Multiple Sclerosis prediction using deep learning.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model cache ───────────────────────────────────────────────
_models = {}


def get_model(disease: str):
    """Lazy-load and cache model."""
    global _models
    if disease not in _models:
        try:
            import tensorflow as tf
            cfg = DISEASE_CONFIG[disease]
            model = tf.keras.models.load_model(cfg["model_path"])
            _models[disease] = model
            print(f"✅ Loaded {disease.upper()} model")
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail=f"Model for '{disease}' not found. Train it first: python src/train.py --disease {disease}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return _models[disease]


def preprocess_bytes(image_bytes: bytes) -> np.ndarray:
    """Preprocess image bytes for model input."""
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def build_response(prob: float, disease: str, threshold: float, filename: str = "") -> dict:
    """Build a structured prediction response."""
    cfg = DISEASE_CONFIG[disease]
    is_positive = prob >= threshold
    predicted_class = cfg["positive_class"] if is_positive else cfg["negative_class"]
    confidence = prob if is_positive else (1.0 - prob)

    if confidence >= 0.85:
        conf_label = "High Confidence"
    elif confidence >= 0.65:
        conf_label = "Moderate Confidence"
    else:
        conf_label = "Low Confidence - Recommend Clinical Review"

    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "disease": cfg["name"],
        "scan_type": cfg["scan_type"],
        "filename": filename,
        "prediction": predicted_class,
        "is_positive": is_positive,
        "probability": round(float(prob), 6),
        "confidence": round(float(confidence), 6),
        "confidence_label": conf_label,
        "threshold": threshold,
        "disclaimer": "This is an AI-assisted tool. Not a substitute for medical diagnosis.",
    }


# ── Pydantic schemas ──────────────────────────────────────────
class Base64Request(BaseModel):
    disease: str
    image_base64: str
    threshold: float = PREDICTION_THRESHOLD

    class Config:
        schema_extra = {
            "example": {
                "disease": "tb",
                "image_base64": "<base64_encoded_image_string>",
                "threshold": 0.5
            }
        }


# ── Routes ────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "TB & MS Medical AI Screening API",
        "version": "1.0.0",
        "supported_diseases": ["tb", "ms"],
        "endpoints": {
            "predict_file": "POST /predict",
            "predict_base64": "POST /predict/base64",
            "health": "GET /health",
            "docs": "GET /docs",
        }
    }


@app.get("/health", tags=["Info"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": list(_models.keys()),
    }


@app.post("/predict", tags=["Prediction"])
async def predict_from_file(
    disease: str = Form(..., description="'tb' or 'ms'"),
    file: UploadFile = File(..., description="X-Ray or MRI image file"),
    threshold: float = Form(PREDICTION_THRESHOLD, description="Classification threshold"),
):
    """
    Predict TB or MS from an uploaded image file.

    - **disease**: 'tb' for Tuberculosis, 'ms' for Multiple Sclerosis
    - **file**: Image file (.jpg, .png, .bmp)
    - **threshold**: Probability threshold (default: 0.5)
    """
    if disease not in ["tb", "ms"]:
        raise HTTPException(status_code=400, detail="Disease must be 'tb' or 'ms'")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        img_array = preprocess_bytes(image_bytes)
        model = get_model(disease)
        prob = float(model.predict(img_array, verbose=0)[0][0])
        return JSONResponse(content=build_response(prob, disease, threshold, file.filename))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/base64", tags=["Prediction"])
async def predict_from_base64(request: Base64Request):
    """
    Predict TB or MS from a base64-encoded image.

    Useful for integrating with frontend apps.
    """
    if request.disease not in ["tb", "ms"]:
        raise HTTPException(status_code=400, detail="Disease must be 'tb' or 'ms'")

    try:
        image_bytes = base64.b64decode(request.image_base64)
        img_array = preprocess_bytes(image_bytes)
        model = get_model(request.disease)
        prob = float(model.predict(img_array, verbose=0)[0][0])
        return JSONResponse(content=build_response(prob, request.disease, request.threshold))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
