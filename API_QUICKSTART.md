# SentinelFS AI - REST API Quick Start Guide

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server

**Option A: Using the startup script**
```bash
./start_api_server.sh
```

**Option B: Using uvicorn directly**
```bash
uvicorn sentinelzer0.api.server:app --reload --port 8000
```

**Option C: Using Python**
```bash
python -m sentinelzer0.api.server
```

The server will start at: `http://localhost:8000`

### 3. Access Documentation

- **Swagger UI**: http://localhost:8000/docs (interactive API testing)
- **ReDoc**: http://localhost:8000/redoc (readable documentation)
- **OpenAPI Schema**: http://localhost:8000/openapi.json

---

## 🔑 Authentication

All protected endpoints require an API key in the `X-API-Key` header.

**Default API Key (Development)**: `sentinelfs-dev-key-2025`

**Example**:
```bash
curl -H "X-API-Key: sentinelfs-dev-key-2025" \
  http://localhost:8000/metrics
```

---

## 📡 API Endpoints

### Public Endpoints (No Auth Required)

#### GET / - API Information
```bash
curl http://localhost:8000/
```

#### GET /health - Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.2.0",
  "model_loaded": true,
  "gpu_available": true,
  "uptime_seconds": 120.5
}
```

---

### Protected Endpoints (API Key Required)

#### POST /predict - Threat Detection

**Single Event**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: sentinelfs-dev-key-2025" \
  -H "Content-Type: application/json" \
  -d '{
    "events": [{
      "event_type": "MODIFY",
      "path": "/home/user/document.txt",
      "timestamp": 1696723200.0,
      "size": 2048,
      "is_directory": false,
      "user": "john",
      "process": "python3",
      "extension": ".txt"
    }],
    "return_components": true
  }'
```

**Batch Events**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: sentinelfs-dev-key-2025" \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {"event_type": "CREATE", "path": "/tmp/file1.txt", "timestamp": 1696723200.0, "size": 1024, "is_directory": false},
      {"event_type": "MODIFY", "path": "/tmp/file2.txt", "timestamp": 1696723201.0, "size": 2048, "is_directory": false},
      {"event_type": "DELETE", "path": "/tmp/file3.txt", "timestamp": 1696723202.0, "size": 512, "is_directory": false}
    ],
    "return_components": false
  }'
```

Response:
```json
{
  "predictions": [
    {
      "event_id": "/home/user/document.txt",
      "timestamp": 1696723200.0,
      "threat_score": 0.12,
      "is_threat": false,
      "confidence": 0.76,
      "latency_ms": 0.85,
      "components": {
        "dl_score": 0.10,
        "if_score": 0.15,
        "heuristic_score": 0.11
      }
    }
  ],
  "total_events": 1,
  "threats_detected": 0,
  "total_latency_ms": 0.85
}
```

#### GET /metrics - Performance Metrics
```bash
curl -H "X-API-Key: sentinelfs-dev-key-2025" \
  http://localhost:8000/metrics
```

Response:
```json
{
  "total_predictions": 10000,
  "threats_detected": 150,
  "threat_rate": 1.5,
  "avg_latency_ms": 0.82,
  "p50_latency_ms": 0.75,
  "p95_latency_ms": 1.2,
  "p99_latency_ms": 2.1,
  "max_latency_ms": 5.3,
  "buffer_size": 64
}
```

#### GET /model/info - Model Information
```bash
curl -H "X-API-Key: sentinelfs-dev-key-2025" \
  http://localhost:8000/model/info
```

#### POST /config/stream - Update Configuration
```bash
curl -X POST "http://localhost:8000/config/stream" \
  -H "X-API-Key: sentinelfs-dev-key-2025" \
  -H "Content-Type: application/json" \
  -d '{
    "threshold": 0.6,
    "min_confidence": 0.7
  }'
```

#### POST /metrics/reset - Reset Metrics
```bash
curl -X POST "http://localhost:8000/metrics/reset" \
  -H "X-API-Key: sentinelfs-dev-key-2025"
```

---

## 🐍 Python Client Example

```python
import requests
import time

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "sentinelfs-dev-key-2025"
headers = {"X-API-Key": API_KEY}

# Check health
response = requests.get(f"{API_URL}/health")
print(f"Status: {response.json()['status']}")

# Analyze file system event
event_data = {
    "events": [{
        "event_type": "MODIFY",
        "path": "/tmp/suspicious_file.exe",
        "timestamp": time.time(),
        "size": 1048576,  # 1MB
        "is_directory": False,
        "user": "admin",
        "process": "unknown.exe",
        "extension": ".exe"
    }],
    "return_components": True
}

response = requests.post(
    f"{API_URL}/predict",
    json=event_data,
    headers=headers
)

result = response.json()
for prediction in result["predictions"]:
    print(f"\nEvent: {prediction['event_id']}")
    print(f"Threat Score: {prediction['threat_score']:.4f}")
    print(f"Is Threat: {prediction['is_threat']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    print(f"Latency: {prediction['latency_ms']:.2f}ms")
    
    if prediction['is_threat']:
        print("⚠️  THREAT DETECTED!")

# Get performance metrics
response = requests.get(f"{API_URL}/metrics", headers=headers)
metrics = response.json()
print(f"\nTotal Predictions: {metrics['total_predictions']}")
print(f"Average Latency: {metrics['avg_latency_ms']:.2f}ms")
```

---

## 🧪 Testing the API

### Run the Test Suite
```bash
# Start the server first
./start_api_server.sh

# In another terminal, run tests
python test_phase_1_2_api.py
```

The test suite will validate:
- ✅ All endpoints
- ✅ Authentication
- ✅ Request/response formats
- ✅ Error handling
- ✅ Performance
- ✅ Documentation

---

## 📊 Event Types

Valid `event_type` values:
- `CREATE` - File/directory creation
- `MODIFY` - File modification
- `DELETE` - File deletion
- `RENAME` - File rename/move
- `CHMOD` - Permission change
- `CHOWN` - Ownership change

---

## ⚡ Performance

- **Latency**: <1ms median per event
- **Throughput**: 1,000+ events/second
- **Concurrency**: Async request handling
- **Batch Processing**: Efficient multi-event analysis

---

## 🔒 Production Deployment

### Security Checklist
- [ ] Change default API key
- [ ] Use environment variables for secrets
- [ ] Enable HTTPS/TLS
- [ ] Configure CORS for specific origins
- [ ] Add rate limiting
- [ ] Implement request logging
- [ ] Set up monitoring and alerts

### Recommended Configuration
```bash
# Use environment variables
export SENTINELFS_API_KEY="your-secure-key-here"
export SENTINELFS_CORS_ORIGINS="https://yourdomain.com"

# Start with production settings
uvicorn sentinelzer0.api.server:app \
  --host 0.0.0.0 \
  --port 443 \
  --ssl-keyfile=/path/to/key.pem \
  --ssl-certfile=/path/to/cert.pem \
  --workers 4 \
  --log-level warning
```

---

## 🐛 Troubleshooting

### Server won't start
```bash
# Check if port 8000 is already in use
lsof -i :8000

# Use a different port
uvicorn sentinelzer0.api.server:app --port 8001
```

### 401 Unauthorized
- Make sure you're including the `X-API-Key` header
- Verify you're using the correct API key

### 503 Service Unavailable
- Check if the model loaded successfully
- Review server logs for errors
- Ensure dependencies are installed

### Slow predictions
- Check GPU availability with `/health`
- Monitor metrics with `/metrics`
- Verify system resources

---

## 📚 Additional Resources

- **Phase 1.1 Summary**: [PHASE_1_1_SUMMARY.md](PHASE_1_1_SUMMARY.md)
- **Phase 1.2 Summary**: [PHASE_1_2_SUMMARY.md](PHASE_1_2_SUMMARY.md)
- **Roadmap**: [ROADMAP.md](ROADMAP.md)

---

## 💬 Support

For issues, questions, or contributions:
- Check the interactive docs at `/docs`
- Review the test suite for examples
- See the summary documents for detailed information

**Version**: 1.2.0  
**Status**: Production Ready ✅
