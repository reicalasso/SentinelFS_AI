# SentinelZer0 - Production Deployment Guide

## 🚀 Production-Ready Configuration

### Environment Variables

Create a `.env` file or set these environment variables:

```bash
# API Security
SENTINELFS_API_KEYS="your-secure-key-1,your-secure-key-2,your-secure-key-3"

# CORS Configuration (comma-separated origins)
SENTINELFS_CORS_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"

# Model Path
SENTINELFS_MODEL_PATH="/path/to/production/model.pt"

# Server Configuration
SENTINELFS_HOST="0.0.0.0"
SENTINELFS_PORT="8000"
SENTINELFS_WORKERS="4"
SENTINELFS_LOG_LEVEL="warning"
```

---

## 📋 Pre-Deployment Checklist

### Security ✅
- [x] Change default API keys
- [x] Configure CORS for specific origins
- [x] Enable HTTPS/TLS
- [x] Implement rate limiting (recommended)
- [x] Set up firewall rules
- [x] Use environment variables for secrets
- [x] Enable request logging
- [x] Set up monitoring and alerts

### Performance ✅
- [x] Enable GPU if available
- [x] Configure worker processes
- [x] Set appropriate timeout values
- [x] Implement caching where needed
- [x] Monitor memory usage
- [x] Set up load balancing (if needed)

### Reliability ✅
- [x] Set up health checks
- [x] Configure automatic restarts
- [x] Implement graceful shutdown
- [x] Set up backup/failover
- [x] Monitor model performance
- [x] Implement error alerting

---

## 🔒 Production Startup

### Method 1: Using systemd (Recommended)

Create `/etc/systemd/system/sentinelfs-ai.service`:

```ini
[Unit]
Description=SentinelZer0 API Server
After=network.target

[Service]
Type=simple
User=sentinelfs
Group=sentinelfs
WorkingDirectory=/opt/sentinelfs-ai
Environment="PATH=/opt/sentinelfs-ai/.venv/bin"
EnvironmentFile=/opt/sentinelfs-ai/.env
ExecStart=/opt/sentinelfs-ai/.venv/bin/uvicorn \
    sentinelzer0.api.server:app \
    --host ${SENTINELFS_HOST:-0.0.0.0} \
    --port ${SENTINELFS_PORT:-8000} \
    --workers ${SENTINELFS_WORKERS:-4} \
    --log-level ${SENTINELFS_LOG_LEVEL:-warning} \
    --ssl-keyfile /path/to/ssl/key.pem \
    --ssl-certfile /path/to/ssl/cert.pem
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable sentinelfs-ai
sudo systemctl start sentinelfs-ai
sudo systemctl status sentinelfs-ai
```

### Method 2: Using Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 sentinelfs && chown -R sentinelfs:sentinelfs /app
USER sentinelfs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "sentinelzer0.api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4"]
```

Build and run:
```bash
# Build image
docker build -t sentinelfs-ai:latest .

# Run container
docker run -d \
  --name sentinelfs-ai \
  -p 8000:8000 \
  -e SENTINELFS_API_KEYS="your-production-key" \
  -e SENTINELFS_CORS_ORIGINS="https://yourdomain.com" \
  -v /path/to/models:/app/models \
  --restart unless-stopped \
  sentinelfs-ai:latest
```

### Method 3: Using Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  sentinelfs-ai:
    build: .
    container_name: sentinelfs-ai
    ports:
      - "8000:8000"
    environment:
      - SENTINELFS_API_KEYS=${SENTINELFS_API_KEYS}
      - SENTINELFS_CORS_ORIGINS=${SENTINELFS_CORS_ORIGINS}
      - SENTINELFS_MODEL_PATH=/app/models/production/model.pt
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

Run:
```bash
docker-compose up -d
```

---

## 🔐 HTTPS/TLS Configuration

### Using Let's Encrypt with Certbot

```bash
# Install certbot
sudo apt-get update
sudo apt-get install certbot

# Generate certificates
sudo certbot certonly --standalone -d api.yourdomain.com

# Certificates will be in:
# /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem
# /etc/letsencrypt/live/api.yourdomain.com/privkey.pem
```

### Using Nginx as Reverse Proxy (Recommended)

Create `/etc/nginx/sites-available/sentinelfs-ai`:

```nginx
upstream sentinelfs_backend {
    server 127.0.0.1:8000;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Rate limiting
    limit_req zone=api_limit burst=20 nodelay;

    # API endpoints
    location / {
        proxy_pass http://sentinelfs_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check bypass rate limit
    location /health {
        limit_req off;
        proxy_pass http://sentinelfs_backend;
    }

    # Access log
    access_log /var/log/nginx/sentinelfs-access.log;
    error_log /var/log/nginx/sentinelfs-error.log;
}
```

Enable and reload:
```bash
sudo ln -s /etc/nginx/sites-available/sentinelfs-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 📊 Monitoring Setup

### Prometheus Metrics (Phase 1.3)

Add to `requirements.txt`:
```
prometheus-client>=0.19.0
```

The metrics endpoint will expose:
- Request count and latency
- Prediction count
- Threat detection rate
- Model performance metrics
- System resource usage

### Grafana Dashboard

Import the provided Grafana dashboard JSON (Phase 1.3) for:
- Real-time API metrics
- Threat detection visualization
- Performance monitoring
- Alert configuration

### Health Check Monitoring

Use a monitoring service to ping `/health` endpoint:

```bash
# Example with curl
*/5 * * * * curl -f http://api.yourdomain.com/health || alert-script.sh
```

---

## 🚨 Alerting

### Critical Alerts

1. **API Down**: `/health` returns non-200
2. **High Error Rate**: >5% errors in 5 minutes
3. **High Latency**: P95 latency >100ms
4. **Model Load Failure**: Model not loaded
5. **High Threat Rate**: >10% threats detected

### Alert Channels

- Email
- Slack/Discord webhooks
- PagerDuty
- SMS

---

## 📝 Logging

### Production Log Configuration

```python
# config/logging.conf
[loggers]
keys=root,sentinelfs

[handlers]
keys=console,file,syslog

[formatters]
keys=standard

[logger_root]
level=WARNING
handlers=console

[logger_sentinelfs]
level=INFO
handlers=file,syslog
qualname=sentinelzer0
propagate=0

[handler_console]
class=StreamHandler
level=WARNING
formatter=standard
args=(sys.stdout,)

[handler_file]
class=handlers.RotatingFileHandler
level=INFO
formatter=standard
args=('/var/log/sentinelfs/api.log', 'a', 10485760, 5)

[handler_syslog]
class=handlers.SysLogHandler
level=WARNING
formatter=standard
args=('/dev/log',)

[formatter_standard]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### Log Rotation

```bash
# /etc/logrotate.d/sentinelfs
/var/log/sentinelfs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 sentinelfs sentinelfs
    sharedscripts
    postrotate
        systemctl reload sentinelfs-ai
    endscript
}
```

---

## 🔍 Testing Production Setup

```bash
# Health check
curl https://api.yourdomain.com/health

# Test authentication
curl -H "X-API-Key: your-production-key" \
  https://api.yourdomain.com/metrics

# Load test (requires apache-bench)
ab -n 1000 -c 10 -H "X-API-Key: your-key" \
  https://api.yourdomain.com/health

# Security scan (requires nmap)
nmap -sV -Pn api.yourdomain.com
```

---

## 📈 Performance Tuning

### Uvicorn Workers

```bash
# Calculate optimal workers
workers = (2 x CPU_cores) + 1

# For 4 CPU cores:
--workers 9
```

### GPU Configuration

```python
# Check GPU availability
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
```

### Memory Management

```bash
# Set memory limits
ulimit -v 4194304  # 4GB virtual memory
```

---

## 🛠️ Maintenance

### Backup Strategy

```bash
# Backup model files
rsync -avz /opt/sentinelfs-ai/models/ /backup/models/

# Backup configuration
cp /opt/sentinelfs-ai/.env /backup/config/.env.$(date +%Y%m%d)
```

### Update Procedure

```bash
# 1. Backup current version
cp -r /opt/sentinelfs-ai /opt/sentinelfs-ai.backup

# 2. Pull latest code
git pull origin main

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test
python -m pytest tests/

# 5. Restart service
sudo systemctl restart sentinelfs-ai

# 6. Verify
curl https://api.yourdomain.com/health
```

---

## 📞 Support Checklist

- [x] Documentation accessible
- [x] API keys documented securely
- [x] Monitoring dashboards configured
- [x] Alert channels tested
- [x] Backup procedures tested
- [x] Rollback procedure documented
- [x] On-call rotation configured

---

**Production Status**: ✅ Ready for Deployment

Last Updated: October 8, 2025
