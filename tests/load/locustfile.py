from locust import HttpUser, task, between
import json
import random

class APIUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Setup test data"""
        self.test_texts = [
            "Kowe ki apik tenan orangnya",
            "Aku seneng banget karo kowe",
            "Sampah kowe ki, ora guna!",
            "Bodoh tenan kowe ki",
            "Sugeng enjing, piye kabare?",
            "Maturnuwun sanget bantuan sampeyan",
            "Ora usah ngomong kaya ngono",
            "Apik banget karyane sampeyan",
            "Aku ora seneng karo kelakuanmu",
            "Selamat pagi, semoga sukses"
        ]
    
    @task(3)
    def test_predict_endpoint(self):
        """Test single prediction - 60% of traffic"""
        text = random.choice(self.test_texts)
        payload = {"text": text}
        
        with self.client.post("/predict", 
                             json=payload,
                             catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "predicted_label" in data and "confidence" in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def test_batch_predict_endpoint(self):
        """Test batch prediction - 40% of traffic"""
        texts = random.sample(self.test_texts, k=random.randint(2, 4))
        payload = {"texts": texts}
        
        with self.client.post("/batch-predict",
                             json=payload,
                             catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "predictions" in data and len(data["predictions"]) == len(texts):
                        response.success()
                    else:
                        response.failure("Invalid batch response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def test_health_endpoint(self):
        """Test health check - 20% of traffic"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "status" in data:
                        response.success()
                    else:
                        response.failure("Invalid health response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def test_model_info_endpoint(self):
        """Test model info endpoint - 20% of traffic"""
        with self.client.get("/model-info", catch_response=True) as response:
            if response.status_code in [200, 503]:  # 503 when model not loaded
                try:
                    data = response.json()
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")

class StressTestUser(HttpUser):
    """High-intensity user for stress testing"""
    wait_time = between(0.1, 0.5)  # Very short wait times
    
    def on_start(self):
        self.test_texts = [
            "Kowe ki apik tenan orangnya",
            "Sampah kowe ki, ora guna!",
            "Sugeng enjing, piye kabare?"
        ]
    
    @task
    def rapid_predict(self):
        """Rapid fire predictions for stress testing"""
        text = random.choice(self.test_texts)
        payload = {"text": text}
        
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

class SpikeTestUser(HttpUser):
    """User for spike testing - simulates sudden traffic spikes"""
    wait_time = between(0.5, 1.0)
    
    def on_start(self):
        self.test_texts = [
            "Kowe ki apik tenan orangnya",
            "Aku seneng banget karo kowe",
            "Sampah kowe ki, ora guna!"
        ]
    
    @task(5)
    def predict_spike(self):
        """Prediction requests during spike"""
        text = random.choice(self.test_texts)
        payload = {"text": text}
        
        self.client.post("/predict", json=payload)
    
    @task(1)
    def health_spike(self):
        """Health checks during spike"""
        self.client.get("/health")