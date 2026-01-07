from pydantic import BaseModel
from typing import List, Optional

class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    text: str

class PredictionResponse(BaseModel):
    """Response model for single prediction"""
    text: str
    predicted_label: str
    confidence: float
    label_id: int
    processing_time: float

class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction"""
    texts: List[str]

class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction"""
    predictions: List[PredictionResponse]
    total_processing_time: float

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    device: str
    model_path: str