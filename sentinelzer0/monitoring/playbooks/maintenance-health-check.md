# System Health Check Playbook

## Overview
This playbook outlines the procedure for performing comprehensive health checks on the SentinelFS AI system to ensure optimal performance and early detection of potential issues.

## Frequency
- **Daily**: Automated health checks
- **Weekly**: Manual comprehensive review
- **Monthly**: Deep-dive analysis
- **Post-Deployment**: After any system changes

## Automated Health Checks

### 1. Service Availability
```bash
# Health endpoint check
curl -f http://localhost:8000/health

# Response time validation
curl -o /dev/null -s -w "%{time_total}\n" http://localhost:8000/health

# API functionality test
curl -f http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"data": "test"}'
```

### 2. Resource Monitoring
```bash
# CPU and memory usage
docker stats sentinelfs-ai --no-stream

# Disk space
df -h /var/lib/docker

# Network connectivity
ping -c 3 <external-dependencies>
```

### 3. Application Metrics
```bash
# Prometheus health check
curl http://localhost:9090/-/healthy

# Key metrics validation
curl http://localhost:9090/api/v1/query?query=up

# Alert status
curl http://localhost:9090/api/v1/alerts
```

## Manual Health Check Procedures

### 1. Application Layer Checks

#### API Endpoints
- [ ] Health endpoint responds (HTTP 200)
- [ ] API endpoints functional
- [ ] Authentication working
- [ ] Rate limiting operational
- [ ] CORS headers correct

#### Model Serving
- [ ] Model loading successful
- [ ] Inference working
- [ ] Confidence scores reasonable
- [ ] Model metrics updating

#### Data Processing
- [ ] Input validation working
- [ ] Data preprocessing functional
- [ ] Output formatting correct
- [ ] Error handling proper

### 2. Infrastructure Layer Checks

#### Containers
- [ ] All containers running
- [ ] Resource limits appropriate
- [ ] Logs clean (no errors)
- [ ] Health checks passing

#### Networking
- [ ] Internal communication working
- [ ] External dependencies accessible
- [ ] Load balancer configuration correct
- [ ] SSL certificates valid

#### Storage
- [ ] Database connectivity
- [ ] Data integrity checks
- [ ] Backup status
- [ ] Log rotation working

### 3. Monitoring Layer Checks

#### Metrics Collection
- [ ] Prometheus scraping working
- [ ] Custom metrics present
- [ ] Metric values reasonable
- [ ] Alert rules functional

#### Logging
- [ ] Logs being written
- [ ] Log levels appropriate
- [ ] Log aggregation working
- [ ] Log retention policies

#### Alerting
- [ ] Alert rules configured
- [ ] Notification channels working
- [ ] Alert thresholds appropriate
- [ ] Alert fatigue not present

## Performance Validation

### 1. Load Testing
```bash
# Basic load test
ab -n 1000 -c 10 http://localhost:8000/api/detect

# Check response times under load
# Verify error rates remain low
# Monitor resource usage
```

### 2. Model Performance
```bash
# Accuracy validation
curl http://localhost:8000/debug/model/accuracy

# Inference time checks
curl http://localhost:8000/debug/model/timing

# Drift detection
curl http://localhost:9090/api/v1/query?query=model_drift_score
```

### 3. Database Performance
```bash
# Connection pool status
# Query performance
# Connection latency
# Data consistency checks
```

## Security Validation

### 1. Access Controls
- [ ] Authentication required for sensitive endpoints
- [ ] Authorization working correctly
- [ ] API keys validated
- [ ] Rate limiting effective

### 2. Data Protection
- [ ] Sensitive data encrypted
- [ ] Logs sanitized
- [ ] Backup security verified
- [ ] Network security configured

### 3. Vulnerability Checks
- [ ] Dependencies up to date
- [ ] Security patches applied
- [ ] Configuration secure
- [ ] Access logs monitored

## Issue Resolution Framework

### Severity Classification
- **Critical**: Service down, data loss, security breach
- **High**: Performance degradation, functionality broken
- **Medium**: Minor issues, monitoring gaps
- **Low**: Optimization opportunities, documentation issues

### Response Times
- **Critical**: Immediate response (< 5 minutes)
- **High**: Response within 1 hour
- **Medium**: Response within 1 day
- **Low**: Address in next maintenance window

### Resolution Steps
1. **Identify**: Gather evidence and symptoms
2. **Assess**: Determine impact and urgency
3. **Plan**: Develop resolution strategy
4. **Execute**: Implement fix safely
5. **Verify**: Confirm resolution and monitor
6. **Document**: Record incident and lessons learned

## Reporting and Documentation

### Health Check Report
```markdown
# SentinelFS AI Health Check Report
Date: [YYYY-MM-DD]
Conducted by: [Name]

## Executive Summary
[Overall system health status]

## Detailed Findings
### Application Layer
- [ ] All checks passed
- Issues found: [List any issues]

### Infrastructure Layer
- [ ] All checks passed
- Issues found: [List any issues]

### Monitoring Layer
- [ ] All checks passed
- Issues found: [List any issues]

## Action Items
1. [Priority] [Description] [Assignee] [Due Date]
2. ...

## Recommendations
[Long-term improvements and optimizations]
```

### Trend Analysis
- Track health check results over time
- Identify recurring issues
- Monitor improvement progress
- Adjust check frequency based on findings

## Automation Opportunities

### Scripted Checks
```bash
#!/bin/bash
# health-check.sh

echo "=== SentinelFS AI Health Check ==="
echo "Timestamp: $(date)"

# Service checks
echo "Checking service availability..."
if curl -f http://localhost:8000/health > /dev/null; then
    echo "✅ Service is healthy"
else
    echo "❌ Service is unhealthy"
fi

# Add more automated checks...
```

### CI/CD Integration
- Health checks in deployment pipeline
- Automated rollback on failures
- Performance regression detection
- Security scanning integration

## Related Playbooks
- [Performance Optimization](maintenance-performance-tuning.md)
- [Backup and Recovery](maintenance-backup-recovery.md)
- [Log Rotation Procedure](maintenance-log-rotation.md)