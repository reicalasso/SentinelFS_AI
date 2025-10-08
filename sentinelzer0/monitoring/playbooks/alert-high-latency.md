# High Latency Alert Response Playbook

## Alert Description
**Alert Name**: High Response Time
**Severity**: Warning (P95 > 1s), Critical (P95 > 2s)
**Description**: API response times have exceeded acceptable thresholds

## Immediate Actions (First 5 minutes)

### 1. Acknowledge Alert
- Acknowledge in AlertManager/Grafana
- Notify on-call engineer via Slack/email
- Update incident tracking system

### 2. Assess Current State
```bash
# Check current metrics
curl http://localhost:9090/api/v1/query?query=http_request_duration_seconds%7Bquantile%3D%220.95%22%7D

# Check system resources
curl http://localhost:8000/health

# View recent logs for errors
tail -n 100 logs/sentinelfs.log | grep -i error
```

### 3. Check System Resources
```bash
# CPU and memory usage
docker stats sentinelfs-ai

# Disk I/O
iostat -x 1 5

# Network connectivity
ping -c 5 <database-host>
ping -c 5 <model-serving-host>
```

## Investigation Steps (5-15 minutes)

### 1. Identify Slow Endpoints
```bash
# Query Prometheus for top slow endpoints
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=topk(5, rate(http_request_duration_seconds_count[5m]))'
```

### 2. Check Database Performance
```bash
# Database connection pool status
# Check for slow queries
# Verify connection limits
```

### 3. Analyze Model Inference Times
```bash
# Check model serving metrics
curl http://localhost:9090/api/v1/query?query=model_inference_duration_seconds

# Verify model server health
curl http://<model-server>:8080/health
```

### 4. Review Recent Changes
- Check deployment history
- Review recent code changes
- Verify configuration changes

## Common Causes and Solutions

### Memory Pressure
**Symptoms**: High memory usage, frequent GC
**Solution**:
```bash
# Scale up memory
docker-compose up -d --scale sentinelfs-ai=2

# Or restart with more memory
docker-compose restart sentinelfs-ai
```

### Database Connection Issues
**Symptoms**: Slow database queries, connection timeouts
**Solution**:
- Check database connection pool settings
- Verify database server resources
- Consider connection pool scaling

### Model Serving Bottleneck
**Symptoms**: High inference times, model server overload
**Solution**:
- Scale model serving instances
- Check model server logs
- Verify GPU/CPU utilization

### Network Latency
**Symptoms**: External API calls slow
**Solution**:
- Check network connectivity
- Verify DNS resolution
- Review external service performance

## Escalation Criteria

### Escalate to Level 2 if:
- Response time > 5 seconds for > 10 minutes
- Multiple endpoints affected
- Customer impact confirmed
- Root cause not identified within 15 minutes

### Escalate to Level 3 if:
- Service becomes unresponsive
- Data loss or corruption suspected
- Security breach indicated
- Widespread system impact

## Recovery Steps

### 1. Implement Immediate Mitigation
- Scale resources horizontally/vertically
- Restart problematic services
- Enable circuit breakers if available

### 2. Monitor Recovery
- Watch metrics for 15 minutes post-mitigation
- Verify alert resolution
- Confirm service stability

### 3. Document Incident
- Record timeline of events
- Document root cause and resolution
- Update monitoring thresholds if needed
- Create follow-up tasks for permanent fixes

## Prevention Measures

### Short-term
- Increase monitoring granularity
- Add more detailed logging
- Implement request tracing

### Long-term
- Performance testing in staging
- Capacity planning reviews
- Code optimization
- Architecture improvements

## Communication Template

**Internal Update**:
```
🚨 High Latency Alert - [Environment]

Status: Investigating
Impact: P95 response time [X]s (threshold [Y]s)
Affected: [List affected endpoints/services]
Actions Taken: [Summary of actions]
Next Steps: [Investigation plan]
ETA: [Estimated resolution time]
```

**Customer Communication** (if needed):
```
We're experiencing some performance issues with our service.
Our team is actively working to resolve this.
Service should be restored within [X] minutes.
We'll provide updates as we have more information.
```

## Related Playbooks
- [Error Rate Spike Response](alert-error-spike.md)
- [Service Unavailability](alert-service-down.md)
- [Performance Optimization](maintenance-performance-tuning.md)