import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_api_info_endpoint():
    """Test API info endpoint"""
    response = client.get("/api")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data

def test_predict_endpoint():
    """Test single prediction endpoint"""
    test_data = {
        "text": "Aku seneng banget karo kowe"
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "predicted_label" in data
    assert "confidence" in data
    assert "label_id" in data
    assert "processing_time" in data

def test_predict_empty_text():
    """Test prediction with empty text"""
    test_data = {
        "text": ""
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 400

def test_batch_predict_endpoint():
    """Test batch prediction endpoint"""
    test_data = {
        "texts": [
            "Aku seneng banget karo kowe",
            "Kowe iku apik tenan",
            "Aku gak seneng karo kowe"
        ]
    }
    response = client.post("/batch-predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "total_processing_time" in data
    assert len(data["predictions"]) == 3

def test_batch_predict_empty_list():
    """Test batch prediction with empty list"""
    test_data = {
        "texts": []
    }
    response = client.post("/batch-predict", json=test_data)
    assert response.status_code == 400

def test_batch_predict_too_many_texts():
    """Test batch prediction with too many texts"""
    test_data = {
        "texts": ["test text"] * 101  # More than 100 texts
    }
    response = client.post("/batch-predict", json=test_data)
    assert response.status_code == 400

if __name__ == "__main__":
    pytest.main([__file__])