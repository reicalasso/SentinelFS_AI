# SentinelFS AI - ELK Stack Setup

This directory contains the complete ELK (Elasticsearch, Logstash, Kibana) stack configuration for centralized logging and monitoring of SentinelFS AI.

## Architecture

```
Application Logs → Filebeat → Logstash → Elasticsearch ← Kibana
                                      ↓
                                 Log Analysis & Dashboards
```

## Components

### Elasticsearch (Port 9200)
- Document storage and search engine
- Stores structured log data with custom mapping
- Single-node configuration for development

### Logstash (Ports 5044, 5000, 9600)
- Log processing and transformation pipeline
- Parses JSON logs, extracts metrics, adds geoip data
- Handles multiple input sources (beats, TCP, UDP, files)

### Kibana (Port 5601)
- Web interface for log visualization and analysis
- Pre-configured dashboards for SentinelFS AI monitoring
- Interactive queries and real-time monitoring

### Filebeat
- Lightweight log shipper
- Monitors log files and forwards to Logstash
- Handles multiline JSON log parsing

## Quick Start

1. **Prerequisites**
   ```bash
   # Ensure Docker and Docker Compose are installed
   docker --version
   docker-compose --version
   ```

2. **Start the ELK Stack**
   ```bash
   cd sentinelfs_ai/monitoring/elk
   docker-compose up -d
   ```

3. **Verify Services**
   ```bash
   # Check service health
   curl -f http://localhost:9200/_cluster/health
   curl -f http://localhost:5601
   curl -f http://localhost:9600
   ```

4. **Import Kibana Dashboard**
   ```bash
   # In Kibana UI, go to Management → Saved Objects
   # Import the sentinelfs-dashboard.ndjson file
   ```

## Configuration Files

- `docker-compose.yml` - Container orchestration
- `logstash/pipeline/logstash.conf` - Log processing pipeline
- `logstash/config/logstash.yml` - Logstash settings
- `logstash/config/sentinelfs-template.json` - Elasticsearch index template
- `filebeat/filebeat.yml` - Filebeat configuration
- `kibana/kibana.yml` - Kibana settings
- `kibana/sentinelfs-dashboard.ndjson` - Pre-built dashboards

## Log Format

SentinelFS AI produces structured JSON logs that include:

```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "INFO",
  "message": "Request processed",
  "request_id": "req-12345",
  "endpoint": "/api/detect",
  "method": "POST",
  "status_code": 200,
  "client_ip": "192.168.1.100",
  "user_agent": "Mozilla/5.0...",
  "metrics": {
    "response_time": 150.5,
    "memory_usage": 512.3,
    "cpu_usage": 15.2,
    "model_inference_time": 120.0,
    "drift_score": 0.05
  },
  "model": {
    "name": "sentinelfs-hybrid",
    "version": "1.0.0",
    "confidence": 0.95
  },
  "environment": "production",
  "version": "3.3.0"
}
```

## Monitoring Dashboards

The included Kibana dashboard provides:

- **Requests Over Time** - Traffic patterns and load monitoring
- **Response Time Distribution** - Performance analysis
- **Error Rate** - System reliability metrics
- **Top Endpoints** - Usage analytics
- **Geographic Distribution** - Request origin analysis
- **Recent Errors** - Error tracking and debugging

## Scaling Considerations

For production deployment:

1. **Elasticsearch Cluster**
   - Configure multiple nodes for high availability
   - Enable security features (xpack.security.enabled=true)
   - Set up proper resource limits

2. **Logstash Pipeline**
   - Increase worker threads for high throughput
   - Add persistent queues for reliability
   - Configure multiple pipeline workers

3. **Monitoring**
   - Enable X-Pack monitoring
   - Set up alerting rules
   - Configure log rotation and retention

## Troubleshooting

### Common Issues

1. **Elasticsearch won't start**
   - Check available memory (minimum 1GB)
   - Verify port 9200 is not in use
   - Check logs: `docker logs sentinelfs_elasticsearch`

2. **Logstash pipeline errors**
   - Validate pipeline configuration
   - Check Logstash logs: `docker logs sentinelfs_logstash`
   - Verify Elasticsearch connectivity

3. **No data in Kibana**
   - Confirm index pattern: `sentinelfs-*`
   - Check Filebeat connectivity to Logstash
   - Verify log file permissions

### Log Locations

- Elasticsearch: `/var/log/elasticsearch/`
- Logstash: `/var/log/logstash/`
- Kibana: `/var/log/kibana/`
- Filebeat: `/var/log/filebeat/`

## Security Notes

This configuration is optimized for development. For production:

- Enable Elasticsearch security
- Configure TLS/SSL certificates
- Set up authentication and authorization
- Use secure networks and firewalls
- Implement log encryption at rest