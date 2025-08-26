import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import time
import random
from fastapi import HTTPException

from app.core.settings import settings
from app.utils.logger import logger

# Global variables
model = None
tokenizer = None
model_loaded = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model path
MODEL_PATH = os.path.join("data", "models", "bert_jawa_hate_speech")

# Demo mode when model is not available
DEMO_MODE = True

def load_model():
    """Load the hate speech detection model"""
    global model, tokenizer, model_loaded
    
    try:
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading model from {MODEL_PATH}...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH,
                num_labels=settings.num_labels
            )
            
            # Move model to device
            model.to(device)
            model.eval()
            
            model_loaded = True
            logger.info(f"Model loaded successfully on {device}")
            return True
            
        else:
            logger.warning(f"Model not found at {MODEL_PATH}. Using demo mode.")
            model_loaded = False
            return False
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded = False
        return False

def predict_text_demo(text: str) -> dict:
    """Demo prediction when model is not available"""
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
        "predicted_label": settings.LABEL_MAPPING[predicted_id] + " (DEMO)",
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
            "predicted_label": settings.LABEL_MAPPING[predicted_id],
            "confidence": confidence,
            "label_id": predicted_id,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error prediksi: {str(e)}")

# Initialize model on import
load_model()