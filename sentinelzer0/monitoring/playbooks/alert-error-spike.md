# Error Rate Spike Alert Response Playbook

## Alert Description
**Alert Name**: High Error Rate
**Severity**: Warning (>2%), Critical (>5%)
**Description**: API error rate has exceeded acceptable thresholds

## Immediate Actions (First 5 minutes)

### 1. Acknowledge Alert
- Acknowledge in AlertManager/Grafana
- Notify on-call engineer
- Create incident ticket

### 2. Assess Current State
```bash
# Check current error rate
curl http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])/rate(http_requests_total[5m])

# Check service health
curl http://localhost:8000/health

# View recent error logs
tail -n 50 logs/sentinelfs.log | grep -i error
```

### 3. Check System Resources
```bash
# Container health
docker ps | grep sentinelfs

# Resource usage
docker stats sentinelfs-ai

# Network connectivity
curl -f http://localhost:8000/api/detect -X POST -d '{}' || echo "Service unreachable"
```

## Investigation Steps (5-15 minutes)

### 1. Identify Error Patterns
```bash
# Top error endpoints
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=topk(5, rate(http_requests_total{status=~"5.."}[5m]))'

# Error types by status code
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=sum(rate(http_requests_total{status=~"5.."}[5m])) by (status)'
```

### 2. Check Application Logs
```bash
# Recent errors in detail
grep -A 5 -B 5 "ERROR" logs/sentinelfs.log | tail -n 100

# Error frequency over time
grep "ERROR" logs/sentinelfs.log | awk '{print $1}' | uniq -c
```

### 3. Database and External Dependencies
```bash
# Database connectivity
# Check external API status
# Verify authentication services
```

### 4. Code and Configuration Review
- Check for recent deployments
- Verify configuration changes
- Review error handling code

## Common Error Scenarios

### 500 Internal Server Error
**Common Causes**:
- Database connection failures
- Model inference errors
- Memory exhaustion
- Code exceptions

**Solutions**:
```bash
# Check application logs for stack traces
tail -f logs/sentinelfs.log | grep -A 10 "Traceback"

# Restart service if needed
docker-compose restart sentinelfs-ai

# Check database connections
```

### 502 Bad Gateway
**Common Causes**:
- Upstream service failures
- Load balancer issues
- Network timeouts

**Solutions**:
- Check upstream service health
- Verify load balancer configuration
- Review network connectivity

### 503 Service Unavailable
**Common Causes**:
- Service overload
- Circuit breaker activation
- Resource exhaustion

**Solutions**:
- Scale service instances
- Check resource limits
- Review circuit breaker settings

### 429 Too Many Requests
**Common Causes**:
- Rate limiting triggered
- DDoS attack
- Client misconfiguration

**Solutions**:
- Review rate limiting rules
- Check client request patterns
- Implement request throttling

## Escalation Criteria

### Escalate to Level 2 if:
- Error rate > 10% for > 5 minutes
- Critical business endpoints affected
- Data loss or corruption indicated
- Customer complaints received

### Escalate to Level 3 if:
- Complete service failure
- Security breach suspected
- Widespread system impact
- Legal or compliance issues

## Recovery Steps

### 1. Implement Fix
- Deploy hotfix if available
- Roll back recent changes
- Scale resources as needed
- Restart services

### 2. Monitor Recovery
- Watch error rate for 10 minutes
- Verify service functionality
- Confirm alert resolution

### 3. Post-Incident Analysis
- Root cause analysis
- Impact assessment
- Prevention measures
- Documentation updates

## Prevention Measures

### Application Level
- Improve error handling
- Add circuit breakers
- Implement graceful degradation
- Enhanced logging and monitoring

### Infrastructure Level
- Auto-scaling policies
- Load balancer improvements
- Database connection pooling
- Caching strategies

### Operational Level
- Regular error rate reviews
- Automated testing
- Configuration management
- Incident response training

## Communication Template

**Internal Update**:
```
🔥 Error Rate Spike Alert - [Environment]

Status: Investigating
Impact: [X]% error rate (threshold [Y]%)
Affected: [List affected endpoints]
Top Errors: [Brief summary]
Actions Taken: [Summary]
Next Steps: [Investigation plan]
ETA: [Estimated resolution]
```

**Customer Communication** (if needed):
```
We're experiencing some technical issues causing intermittent errors.
Our engineering team is actively resolving this.
Most requests should work normally.
We'll provide updates as the situation improves.
```

## Related Playbooks
- [High Latency Response](alert-high-latency.md)
- [Service Unavailability](alert-service-down.md)
- [Security Incident Response](incident-security-breach.md)