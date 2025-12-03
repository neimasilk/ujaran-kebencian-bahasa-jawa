from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import os
from typing import List, Optional
import time

from app.core.settings import settings
from app.utils.logger import logger

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="Javanese Hate Speech Detection API",
    description="API untuk deteksi ujaran kebencian dalam Bahasa Jawa menggunakan IndoBERT",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global variables for model and tokenizer
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_loaded = False

# Constants
MODEL_PATH = os.path.join("data", "models", "bert_jawa_hate_speech")
LABEL_MAPPING = settings.LABEL_MAPPING

# Demo mode when model is not available
DEMO_MODE = True

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    text: str
    
class BatchPredictionRequest(BaseModel):
    texts: List[str]
    
class PredictionResponse(BaseModel):
    text: str
    predicted_label: str
    confidence: float
    label_id: int
    processing_time: float
    
class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processing_time: float
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_path: str

def load_model():
    """Load model and tokenizer"""
    global model, tokenizer, model_loaded
    
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model path tidak ditemukan: {MODEL_PATH}")
            return False
            
        logger.info(f"Loading model dari {MODEL_PATH}...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        model.to(device)
        model.eval()
        
        model_loaded = True
        logger.info(f"Model berhasil dimuat pada device: {device}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded = False
        return False

def predict_text_demo(text: str) -> dict:
    """Demo prediction when model is not available"""
    import random
    start_time = time.time()
    
    # Simple rule-based demo logic
    text_lower = text.lower()
    hate_keywords = ['benci', 'bodoh', 'tolol', 'goblok', 'anjing', 'bangsat', 'sialan']
    
    # Check for hate speech keywords
    hate_score = sum(1 for keyword in hate_keywords if keyword in text_lower)
    
    if hate_score >= 2:
        predicted_id = 3  # Berat
        confidence = 0.85 + random.uniform(0, 0.1)
    elif hate_score == 1:
        predicted_id = random.choice([1, 2])  # Ringan atau Sedang
        confidence = 0.70 + random.uniform(0, 0.15)
    else:
        predicted_id = 0  # Bukan ujaran kebencian
        confidence = 0.80 + random.uniform(0, 0.15)
    
    processing_time = time.time() - start_time
    
    return {
        "text": text,
        "predicted_label": LABEL_MAPPING[predicted_id] + " (DEMO)",
        "confidence": min(confidence, 0.99),
        "label_id": predicted_id,
        "processing_time": processing_time
    }

def predict_text(text: str) -> dict:
    """Predict single text"""
    if not model_loaded or model is None or tokenizer is None:
        if DEMO_MODE:
            return predict_text_demo(text)
        else:
            raise HTTPException(status_code=503, detail="Model belum dimuat")
    
    start_time = time.time()
    
    try:
        # Tokenize input
        inputs = tokenizer(
            text, 
            truncation=True, 
            padding=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_id = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()
        
        processing_time = time.time() - start_time
        
        return {
            "text": text,
            "predicted_label": LABEL_MAPPING[predicted_id],
            "confidence": confidence,
            "label_id": predicted_id,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error prediksi: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Javanese Hate Speech Detection API...")
    success = load_model()
    if not success:
        logger.warning("Model gagal dimuat saat startup. API akan tetap berjalan tapi prediksi tidak tersedia.")

@app.get("/", response_class=FileResponse)
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

@app.get("/api", response_model=dict)
async def api_info():
    """API info endpoint"""
    return {
        "message": "Javanese Hate Speech Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_loaded
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        device=str(device),
        model_path=MODEL_PATH
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Predict single text"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text tidak boleh kosong")
    
    result = predict_text(request.text)
    return PredictionResponse(**result)

@app.post("/batch-predict", response_model=BatchPredictionResponse)
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

@app.post("/reload-model")
async def reload_model():
    """Reload model endpoint"""
    success = load_model()
    if success:
        return {"message": "Model berhasil dimuat ulang", "status": "success"}
    else:
        raise HTTPException(status_code=500, detail="Gagal memuat ulang model")

@app.get("/model-info")
async def model_info():
    """Get model information"""
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
            "labels": LABEL_MAPPING,
            "max_sequence_length": 128
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)