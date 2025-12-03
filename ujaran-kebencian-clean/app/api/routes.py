from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List
import os
import time

from app.api.models import (
    PredictionRequest, 
    PredictionResponse, 
    BatchPredictionRequest, 
    BatchPredictionResponse,
    HealthResponse
)
from app.core.settings import settings
from app.utils.logger import logger

# Import prediction functions (will be implemented)
from app.models.predict import predict_text, model_loaded, device, MODEL_PATH

router = APIRouter()

@router.get("/", response_class=FileResponse)
async def root():
    """Serve web interface"""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    index_path = os.path.join(static_dir, "index.html")
    
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        # Fallback to JSON response if no web interface
        return {
            "message": "Javanese Hate Speech Detection API",
            "version": "1.0.0",
            "status": "running",
            "model_loaded": model_loaded,
            "note": "Web interface not found. Use /docs for API documentation."
        }

@router.get("/api", response_model=dict)
async def api_info():
    """API info endpoint"""
    return {
        "message": "Javanese Hate Speech Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_loaded
    }

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        device=str(device),
        model_path=MODEL_PATH
    )

@router.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Predict single text"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text tidak boleh kosong")
    
    result = predict_text(request.text)
    return PredictionResponse(**result)

@router.post("/batch-predict", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict multiple texts"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="List texts tidak boleh kosong")
    
    if len(request.texts) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maksimal 100 teks per batch")
    
    start_time = time.time()
    predictions = []
    
    for text in request.texts:
        if text.strip():  # Skip empty texts
            try:
                result = predict_text(text)
                predictions.append(PredictionResponse(**result))
            except Exception as e:
                logger.error(f"Error predicting text '{text[:50]}...': {e}")
                # Add error prediction
                predictions.append(PredictionResponse(
                    text=text,
                    predicted_label="Error",
                    confidence=0.0,
                    label_id=-1,
                    processing_time=0.0
                ))
    
    total_time = time.time() - start_time
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_processing_time=total_time
    )

@router.post("/reload-model")
async def reload_model():
    """Reload model endpoint"""
    from app.models.predict import load_model
    success = load_model()
    if success:
        return {"message": "Model berhasil dimuat ulang", "status": "success"}
    else:
        raise HTTPException(status_code=500, detail="Gagal memuat ulang model")

@router.get("/model-info")
async def model_info():
    """Get model information"""
    from app.models.predict import model
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model belum dimuat")
    
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "model_path": MODEL_PATH,
            "device": str(device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": round(total_params * 4 / 1024**2, 2),
            "labels": settings.LABEL_MAPPING,
            "max_sequence_length": 128
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")