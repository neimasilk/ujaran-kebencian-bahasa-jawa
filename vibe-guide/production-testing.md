# Production Testing & Monitoring Guide

## Overview
This document outlines production testing strategies, monitoring procedures, and performance benchmarks for the Javanese Hate Speech Detection API in production environments.

## Load Testing

### Testing Framework
We use `locust` for load testing to simulate realistic user traffic patterns.

### Installation
```bash
pip install locust
```

### Load Test Configuration

#### Basic Load Test Script
```python
# tests/load/locustfile.py
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
            "Sugeng enjing, piye kabare?"
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
                data = response.json()
                if "predicted_label" in data and "confidence" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
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
                data = response.json()
                if "predictions" in data and len(data["predictions"]) == len(texts):
                    response.success()
                else:
                    response.failure("Invalid batch response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def test_health_endpoint(self):
        """Test health check - 20% of traffic"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "status" in data:
                    response.success()
                else:
                    response.failure("Invalid health response")
            else:
                response.failure(f"Status code: {response.status_code}")
```

### Running Load Tests

#### Local Testing
```bash
# Start API server
cd src/api
python run_server.py

# Run load test (separate terminal)
cd tests/load
locust -f locustfile.py --host=http://localhost:8000
```

#### Production Testing
```bash
# Test production endpoint
locust -f locustfile.py --host=https://your-api-domain.com
```

### Load Test Scenarios

#### 1. Normal Load
- **Users**: 10-50 concurrent users
- **Duration**: 10 minutes
- **Expected**: < 200ms response time, 0% error rate

#### 2. Peak Load
- **Users**: 100-200 concurrent users
- **Duration**: 5 minutes
- **Expected**: < 500ms response time, < 1% error rate

#### 3. Stress Test
- **Users**: 500+ concurrent users
- **Duration**: 2 minutes
- **Expected**: Graceful degradation, no crashes

#### 4. Spike Test
- **Pattern**: Sudden increase from 10 to 200 users
- **Duration**: 1 minute spike
- **Expected**: System recovery within 30 seconds

## Performance Benchmarks

### Response Time Targets
| Endpoint | Target (95th percentile) | Maximum Acceptable |
|----------|-------------------------|-------------------|
| `/predict` | < 200ms | < 500ms |
| `/batch-predict` | < 500ms | < 1000ms |
| `/health` | < 50ms | < 100ms |
| `/model-info` | < 100ms | < 200ms |

### Throughput Targets
- **Single Predictions**: 100+ requests/second
- **Batch Predictions**: 50+ requests/second
- **Health Checks**: 500+ requests/second

### Resource Utilization
- **CPU**: < 80% under normal load
- **Memory**: < 2GB for model + API
- **Disk I/O**: < 50% utilization
- **Network**: < 100 Mbps

## Monitoring Setup

### Application Metrics

#### Custom Metrics to Track
```python
# Add to main.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration', ['endpoint'])
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total predictions made', ['label'])
ACTIVE_CONNECTIONS = Gauge('api_active_connections', 'Active connections')
MODEL_LOAD_TIME = Gauge('model_load_time_seconds', 'Model loading time')

# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_DURATION.labels(endpoint=request.url.path).observe(duration)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response
```

#### Health Check Enhancements
```python
@app.get("/health/detailed")
async def detailed_health():
    """Detailed health check with metrics"""
    import psutil
    import torch
    
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "device": str(device),
        "model_path": MODEL_PATH,
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        },
        "api": {
            "uptime_seconds": time.time() - startup_time,
            "total_requests": REQUEST_COUNT._value.sum(),
            "active_connections": ACTIVE_CONNECTIONS._value.get()
        },
        "timestamp": datetime.now().isoformat()
    }
```

### Logging Configuration

#### Structured Logging
```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration
            
        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('logs/api.log'))
for handler in logger.handlers:
    handler.setFormatter(JSONFormatter())
```

### Alerting Rules

#### Critical Alerts
- **API Down**: Health check fails for > 1 minute
- **High Error Rate**: > 5% error rate for > 2 minutes
- **High Response Time**: 95th percentile > 1 second for > 5 minutes
- **Memory Usage**: > 90% for > 5 minutes
- **CPU Usage**: > 95% for > 3 minutes

#### Warning Alerts
- **Moderate Error Rate**: > 1% error rate for > 5 minutes
- **Slow Response**: 95th percentile > 500ms for > 10 minutes
- **High Memory**: > 80% for > 10 minutes
- **Model Not Loaded**: Model fails to load

## Production Deployment Checklist

### Pre-Deployment
- [ ] All tests pass (unit, integration, load)
- [ ] Security scan completed
- [ ] Performance benchmarks met
- [ ] Monitoring configured
- [ ] Alerting rules set up
- [ ] Backup procedures tested
- [ ] Rollback plan prepared

### Deployment
- [ ] Blue-green deployment strategy
- [ ] Health checks pass
- [ ] Smoke tests completed
- [ ] Performance validation
- [ ] Monitor error rates
- [ ] Verify all endpoints

### Post-Deployment
- [ ] Monitor for 24 hours
- [ ] Check all alerts
- [ ] Validate performance metrics
- [ ] Review error logs
- [ ] Update documentation
- [ ] Notify stakeholders

## Incident Response

### Severity Levels

#### P0 - Critical
- API completely down
- Data corruption
- Security breach
- **Response Time**: 15 minutes

#### P1 - High
- Significant performance degradation
- High error rates
- Feature unavailable
- **Response Time**: 1 hour

#### P2 - Medium
- Minor performance issues
- Non-critical feature issues
- **Response Time**: 4 hours

#### P3 - Low
- Documentation issues
- Minor bugs
- **Response Time**: 24 hours

### Response Procedures

1. **Acknowledge**: Confirm incident within SLA
2. **Assess**: Determine severity and impact
3. **Communicate**: Notify stakeholders
4. **Investigate**: Identify root cause
5. **Mitigate**: Implement temporary fix
6. **Resolve**: Deploy permanent solution
7. **Review**: Post-incident analysis

## Maintenance Windows

### Scheduled Maintenance
- **Frequency**: Monthly
- **Duration**: 2 hours
- **Time**: Sunday 2-4 AM UTC
- **Activities**:
  - Security updates
  - Performance optimization
  - Model updates
  - Infrastructure maintenance

### Emergency Maintenance
- **Trigger**: Critical security issues
- **Approval**: Required from team lead
- **Communication**: 1 hour advance notice
- **Rollback**: Ready within 15 minutes

---

**Last Updated**: December 2024  
**Version**: 1.0  
**Maintainer**: DevOps & Backend Team