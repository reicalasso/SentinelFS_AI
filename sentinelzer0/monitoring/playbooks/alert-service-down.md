# Service Unavailability Alert Response Playbook

## Alert Description
**Alert Name**: Service Down / Unavailable
**Severity**: Critical
**Description**: SentinelFS AI service is not responding to requests

## Immediate Actions (First 2 minutes)

### 1. Acknowledge Alert
- Acknowledge in monitoring system
- Notify on-call engineer immediately
- Alert incident response team

### 2. Confirm Outage
```bash
# Check service availability
curl -f --max-time 5 http://localhost:8000/health || echo "Service DOWN"

# Check container status
docker ps | grep sentinelfs

# Check application logs
tail -n 10 logs/sentinelfs.log
```

### 3. Assess Impact
- Check affected endpoints
- Estimate user impact
- Determine business criticality

## Investigation Steps (2-10 minutes)

### 1. Check Infrastructure Health
```bash
# Docker container health
docker inspect sentinelfs-ai | grep -A 5 "State"

# System resources
docker stats sentinelfs-ai --no-stream

# Network connectivity
ping -c 3 localhost
```

### 2. Review Application Logs
```bash
# Recent errors
tail -n 50 logs/sentinelfs.log | grep -i error

# Startup/shutdown events
grep -i "start\|stop\|shutdown\|crash" logs/sentinelfs.log | tail -n 20
```

### 3. Check Dependencies
```bash
# Database connectivity
# Model serving health
# External API status
# Load balancer status
```

### 4. Verify Configuration
- Check environment variables
- Verify configuration files
- Confirm resource limits

## Common Failure Scenarios

### Application Crash
**Symptoms**: Container running but unresponsive
**Solutions**:
```bash
# Check application process
docker exec sentinelfs-ai ps aux | grep python

# View application logs
docker logs sentinelfs-ai --tail 50

# Restart application
docker-compose restart sentinelfs-ai
```

### Container Failure
**Symptoms**: Container stopped or restarting
**Solutions**:
```bash
# Check container logs
docker logs sentinelfs-ai

# Restart container
docker-compose up -d sentinelfs-ai

# Check resource constraints
docker system df
```

### Resource Exhaustion
**Symptoms**: High CPU/memory usage, OOM kills
**Solutions**:
```bash
# Check resource usage
docker stats sentinelfs-ai

# Scale resources
docker-compose up -d --scale sentinelfs-ai=2

# Or increase limits
echo "Increasing memory limit..."
```

### Network Issues
**Symptoms**: Service accessible locally but not externally
**Solutions**:
- Check load balancer
- Verify firewall rules
- Check DNS resolution
- Review network configuration

### Database Connectivity
**Symptoms**: Service starts but fails on data operations
**Solutions**:
- Check database server status
- Verify connection strings
- Test database connectivity
- Check connection pool

## Recovery Procedures

### Quick Recovery (5 minutes)
1. **Restart Service**
   ```bash
   docker-compose restart sentinelfs-ai
   ```

2. **Scale Resources**
   ```bash
   docker-compose up -d --scale sentinelfs-ai=2
   ```

3. **Check Health**
   ```bash
   curl http://localhost:8000/health
   ```

### Full Recovery (15-30 minutes)
1. **Investigate Root Cause**
   - Analyze crash logs
   - Check system resources
   - Review recent changes

2. **Implement Fix**
   - Apply configuration changes
   - Update resource limits
   - Deploy code fixes

3. **Validate Recovery**
   - Monitor for 10 minutes
   - Run health checks
   - Verify functionality

## Escalation Criteria

### Escalate Immediately if:
- Service down for > 5 minutes
- Multiple services affected
- Data loss suspected
- Security breach indicated

### Escalate to Level 3 if:
- Manual intervention required
- Infrastructure failure
- Vendor dependency issues
- Widespread impact

## Communication Plan

### Internal Communication
**Immediate Alert**:
```
🚨 CRITICAL: SentinelFS AI Service DOWN

Status: Investigating
Impact: Complete service outage
Affected: All API endpoints
Started: [Timestamp]
Actions: Restarting service, checking logs
Team: @oncall-engineers
```

**Updates Every 5 Minutes**:
```
Service Status Update:
Time: [Timestamp]
Status: [Investigating/Recovering/Resolved]
Current Actions: [What we're doing]
ETA: [Estimated recovery time]
```

### External Communication
**Customer Notification** (if outage > 5 minutes):
```
Subject: SentinelFS AI Service Interruption

Dear Customer,

We're currently experiencing a service interruption affecting SentinelFS AI.
Our team is working to restore service as quickly as possible.

Current Status: Investigating
Estimated Resolution: [ETA]
Impact: [Description of impact]

We'll provide updates as we have more information.
Thank you for your patience.

Best regards,
SentinelFS AI Team
```

**Status Page Update**:
- Update service status page
- Post incident details
- Provide timeline and resolution

## Post-Incident Activities

### 1. Root Cause Analysis
- Detailed timeline reconstruction
- Code review for bugs
- Infrastructure review
- Process improvement identification

### 2. Documentation Updates
- Update incident runbook
- Document new failure modes
- Update monitoring alerts
- Create knowledge base articles

### 3. Prevention Measures
- Implement additional monitoring
- Add automated recovery
- Improve testing procedures
- Update deployment processes

## Prevention Measures

### Application Level
- Add health checks and readiness probes
- Implement graceful shutdown
- Add circuit breakers
- Improve error handling

### Infrastructure Level
- Auto-scaling policies
- Load balancer health checks
- Backup and failover systems
- Resource monitoring

### Operational Level
- Regular failover testing
- Incident response training
- Monitoring coverage review
- Deployment rollback procedures

## Related Playbooks
- [High Latency Response](alert-high-latency.md)
- [Error Rate Spike Response](alert-error-spike.md)
- [Backup and Recovery](maintenance-backup-recovery.md)