# 🚀 SentinelZer0 Quick Start

## Prerequisites

- Docker & Docker Compose
- 8GB RAM minimum
- Python 3.10+ (for local development)

## 1. Quick Deploy (Recommended)

```bash
# Clone and enter directory
cd SentınelFS_AI

# Start all services
./quick-start.sh

# Or manually:
docker-compose up -d
```

## 2. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **API** | http://localhost:8000 | - |
| **API Docs** | http://localhost:8000/docs | - |
| **Grafana** | http://localhost:3000 | admin / sentinelzer0 |
| **Prometheus** | http://localhost:9091 | - |
| **MLflow** | http://localhost:5000 | - |

## 3. Test API

```bash
# Health check
curl http://localhost:8000/health

# Predict threat
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "events": [{
      "event_type": "file_write",
      "path": "/etc/passwd",
      "process": "malicious.exe"
    }]
  }'
```

## 4. Monitor

- **Logs**: `docker-compose logs -f sentinelzer0`
- **Status**: `docker-compose ps`
- **Metrics**: http://localhost:9091/graph

## 5. Stop Services

```bash
docker-compose down
```

## 📚 Full Documentation

- [README.md](README.md) - Complete guide
- [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md) - Deployment checklist
- [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) - Production guide
- [CHANGELOG.md](CHANGELOG.md) - Version history

## 🆘 Troubleshooting

**Port conflicts:**
```bash
# Edit docker-compose.yml and change ports
# Default ports: 8000, 3000, 5000, 6380, 9091
```

**Container won't start:**
```bash
docker-compose logs sentinelzer0
```

**Reset everything:**
```bash
docker-compose down -v
docker-compose up -d --build
```

## 🛠️ Development

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run tests
pytest tests/

# Local API (without Docker)
python -m uvicorn sentinelzer0.api.server:app --reload
```

## 📊 Performance

- **Inference**: < 5ms per event
- **Throughput**: 10,000+ events/second
- **Accuracy**: 95%+ on test data
- **False Positive Rate**: < 1%

---

**Version**: 3.8.0  
**Status**: Production Ready ✅
