# 🔑 SentinelZer0 API Cheat Sheet

## 🚀 Quick Access

### API Key (Development)
```
X-API-Key: sentinelfs-dev-key-2025
```

### Endpoints
```
Health:      http://localhost:8000/health
API Docs:    http://localhost:8000/docs
Metrics:     http://localhost:8000/metrics
Grafana:     http://localhost:3000 (admin/sentinelzer0)
Prometheus:  http://localhost:9091
```

---

## 📡 API Examples

### 1. Health Check (No Auth)
```bash
curl http://localhost:8000/health | jq .
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.2.0",
  "model_loaded": true,
  "gpu_available": false,
  "uptime_seconds": 123.45
}
```

---

### 2. Single Event Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sentinelfs-dev-key-2025" \
  -d '{
    "events": [{
      "timestamp": 1728403200,
      "event_type": "MODIFY",
      "path": "/etc/shadow",
      "process_name": "unknown.exe",
      "user": "www-data",
      "file_size": 4096
    }]
  }' | jq .
```

---

### 3. Ransomware Simulation (Multiple Events)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sentinelfs-dev-key-2025" \
  -d '{
    "events": [
      {
        "timestamp": 1728403200,
        "event_type": "CREATE",
        "path": "/home/user/document.txt.encrypted",
        "process_name": "ransomware.exe",
        "user": "user",
        "file_size": 5120
      },
      {
        "timestamp": 1728403201,
        "event_type": "DELETE",
        "path": "/home/user/document.txt",
        "process_name": "ransomware.exe",
        "user": "user",
        "file_size": 0
      },
      {
        "timestamp": 1728403202,
        "event_type": "CREATE",
        "path": "/home/user/README_RANSOM.txt",
        "process_name": "ransomware.exe",
        "user": "user",
        "file_size": 512
      }
    ]
  }' | jq .
```

---

### 4. Stream Processing
```bash
curl -X POST http://localhost:8000/stream/events \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sentinelfs-dev-key-2025" \
  -d '{
    "events": [
      {
        "timestamp": 1728403200,
        "event_type": "MODIFY",
        "path": "/etc/passwd",
        "process_name": "malware.exe",
        "user": "root",
        "file_size": 2048
      }
    ]
  }' | jq .
```

---

### 5. Get Configuration
```bash
curl -X GET http://localhost:8000/config \
  -H "X-API-Key: sentinelfs-dev-key-2025" | jq .
```

---

### 6. Update Configuration
```bash
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sentinelfs-dev-key-2025" \
  -d '{
    "threshold": 0.7,
    "sequence_length": 64
  }' | jq .
```

---

### 7. Get Metrics
```bash
curl http://localhost:8000/metrics
```

---

## 📊 Event Types (Must Use These!)

```python
"CREATE"   # File/directory creation
"MODIFY"   # File modification/write
"DELETE"   # File/directory deletion
"RENAME"   # File/directory rename
"CHMOD"    # Permission change
"CHOWN"    # Owner change
```

---

## 🔑 Event Schema

### Required Fields:
```json
{
  "timestamp": 1728403200,        // Unix timestamp
  "event_type": "MODIFY",         // See Event Types above
  "path": "/path/to/file",        // File path
  "process_name": "process.exe",  // Process name
  "user": "username",             // User name
  "file_size": 4096              // File size in bytes
}
```

### Optional Fields:
```json
{
  "pid": 1234,                    // Process ID
  "ppid": 1000,                   // Parent process ID
  "extension": ".exe",            // File extension
  "is_system": false,             // System file flag
  "is_hidden": false,             // Hidden file flag
  "permissions": 644              // File permissions
}
```

---

## 🧪 Test Scenarios

### Scenario 1: Normal Activity
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sentinelfs-dev-key-2025" \
  -d '{
    "events": [{
      "timestamp": 1728403200,
      "event_type": "CREATE",
      "path": "/home/user/myfile.txt",
      "process_name": "notepad.exe",
      "user": "user",
      "file_size": 1024
    }]
  }' | jq .
```

### Scenario 2: Suspicious System File Access
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sentinelfs-dev-key-2025" \
  -d '{
    "events": [{
      "timestamp": 1728403200,
      "event_type": "MODIFY",
      "path": "/etc/shadow",
      "process_name": "unknown.bin",
      "user": "www-data",
      "file_size": 4096,
      "is_system": true
    }]
  }' | jq .
```

### Scenario 3: Mass File Encryption (Ransomware)
```bash
# Use the ransomware simulation example above
```

---

## 🐛 Debugging

### Check Container Logs
```bash
docker-compose logs -f sentinelzer0
```

### Check All Services
```bash
docker-compose ps
```

### Restart API
```bash
docker-compose restart sentinelzer0
```

### Check API Health
```bash
curl http://localhost:8000/health
```

### View Metrics
```bash
curl http://localhost:8000/metrics | grep -E "sentinelfs|prediction"
```

---

## 🎯 Common Issues

### "Missing API key"
**Solution:** Add header `-H "X-API-Key: sentinelfs-dev-key-2025"`

### "Invalid event_type"
**Solution:** Use: CREATE, MODIFY, DELETE, RENAME, CHMOD, CHOWN

### "Connection refused"
**Solution:** Check if services are running:
```bash
docker-compose ps
docker-compose logs sentinelzer0
```

### No threats detected
**Note:** Model needs training with real threat data. Currently using randomly initialized weights.

---

## 📚 Interactive API Docs

Visit: **http://localhost:8000/docs**

- ✅ Try all endpoints interactively
- ✅ See request/response schemas
- ✅ Built-in authentication
- ✅ Download OpenAPI spec

---

## 🔄 Environment Variables

Create `.env` file:
```bash
# API Configuration
SENTINELFS_API_KEYS=your-custom-key-1,your-custom-key-2
SENTINELFS_CORS_ORIGINS=http://localhost:3000
SENTINELFS_MODEL_PATH=models/production/your_model.pt

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Performance
API_WORKERS=4
MODEL_DEVICE=cuda
```

Then restart:
```bash
docker-compose down
docker-compose up -d
```

---

**Version:** 3.8.0  
**Last Updated:** Oct 8, 2025
