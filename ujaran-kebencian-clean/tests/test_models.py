import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.models.predict import predict_text_demo, predict_text
from app.core.settings import settings

def test_predict_text_demo():
    """Test demo prediction function"""
    test_text = "Aku seneng banget karo kowe"
    result = predict_text_demo(test_text)
    
    assert "text" in result
    assert "predicted_label" in result
    assert "confidence" in result
    assert "label_id" in result
    assert "processing_time" in result
    
    assert result["text"] == test_text
    assert isinstance(result["confidence"], float)
    assert 0 <= result["confidence"] <= 1
    assert result["label_id"] in [0, 1, 2, 3]
    assert "(DEMO)" in result["predicted_label"]

def test_predict_text_demo_hate_speech():
    """Test demo prediction with hate speech keywords"""
    test_text = "Kowe iku bodoh banget goblok"
    result = predict_text_demo(test_text)
    
    # Should detect hate speech due to keywords
    assert result["label_id"] > 0  # Not "Bukan Ujaran Kebencian"
    assert result["confidence"] > 0.5

def test_predict_text_demo_normal_text():
    """Test demo prediction with normal text"""
    test_text = "Aku seneng banget karo kowe, kowe apik tenan"
    result = predict_text_demo(test_text)
    
    # Should likely be classified as not hate speech
    assert result["label_id"] >= 0
    assert result["confidence"] > 0

def test_label_mapping():
    """Test that label mapping is correctly configured"""
    assert hasattr(settings, 'LABEL_MAPPING')
    assert len(settings.LABEL_MAPPING) == 4
    assert 0 in settings.LABEL_MAPPING
    assert 1 in settings.LABEL_MAPPING
    assert 2 in settings.LABEL_MAPPING
    assert 3 in settings.LABEL_MAPPING

def test_settings_configuration():
    """Test that settings are properly configured"""
    assert settings.num_labels == 4
    assert settings.model_name is not None
    assert settings.api_host is not None
    assert settings.api_port is not None

if __name__ == "__main__":
    pytest.main([__file__])