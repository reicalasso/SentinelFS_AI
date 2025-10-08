"""
Alerting System for SentinelFS AI

This module provides configurable alerting for various system events
and performance issues.
"""

import logging
import time
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import json

from .metrics import record_alert

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of alerts that can be triggered."""
    DRIFT = "drift"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    MEMORY = "memory"
    PREDICTION_FAILURE = "prediction_failure"
    SYSTEM_HEALTH = "system_health"

@dataclass
class Alert:
    """Alert data structure."""
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: float
    metadata: Dict
    resolved: bool = False
    resolved_at: Optional[float] = None

class AlertManager:
    """
    Manages alerts and notifications for the SentinelFS AI system.

    Features:
    - Configurable alert rules
    - Alert deduplication
    - Multiple notification channels
    - Alert history and resolution tracking
    """

    def __init__(self):
        """Initialize the alert manager."""
        self.alerts = []
        self.alert_rules = []
        self.notification_handlers = []
        self.alert_history = []
        self.max_history_size = 1000

        # Default alert rules
        self._setup_default_rules()

        logger.info("Alert manager initialized")

    def _setup_default_rules(self):
        """Set up default alert rules."""
        # High latency alert
        self.add_rule(
            name="high_latency",
            type=AlertType.LATENCY,
            severity=AlertSeverity.WARNING,
            condition=lambda metrics: metrics.get('avg_latency', 0) > 1.0,
            message="Average request latency exceeded 1 second",
            cooldown=300  # 5 minutes
        )

        # Critical latency alert
        self.add_rule(
            name="critical_latency",
            type=AlertType.LATENCY,
            severity=AlertSeverity.CRITICAL,
            condition=lambda metrics: metrics.get('avg_latency', 0) > 5.0,
            message="Critical: Average request latency exceeded 5 seconds",
            cooldown=60  # 1 minute
        )

        # High error rate alert
        self.add_rule(
            name="high_error_rate",
            type=AlertType.ERROR_RATE,
            severity=AlertSeverity.ERROR,
            condition=lambda metrics: metrics.get('error_rate', 0) > 0.05,  # 5% error rate
            message="Error rate exceeded 5%",
            cooldown=300
        )

        # Memory usage alert
        self.add_rule(
            name="high_memory",
            type=AlertType.MEMORY,
            severity=AlertSeverity.WARNING,
            condition=lambda metrics: metrics.get('memory_percent', 0) > 90.0,
            message="Memory usage exceeded 90%",
            cooldown=300
        )

        # Model drift alert
        self.add_rule(
            name="model_drift",
            type=AlertType.DRIFT,
            severity=AlertSeverity.WARNING,
            condition=lambda metrics: metrics.get('drift_score', 0) > 0.1,
            message="Model drift detected",
            cooldown=600  # 10 minutes
        )

    def add_rule(
        self,
        name: str,
        type: AlertType,
        severity: AlertSeverity,
        condition: Callable[[Dict], bool],
        message: str,
        cooldown: int = 300
    ):
        """
        Add a custom alert rule.

        Args:
            name: Unique name for the rule
            type: Type of alert
            severity: Severity level
            condition: Function that takes metrics dict and returns bool
            message: Alert message
            cooldown: Minimum seconds between alerts of this type
        """
        rule = {
            'name': name,
            'type': type,
            'severity': severity,
            'condition': condition,
            'message': message,
            'cooldown': cooldown,
            'last_triggered': 0
        }
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {name}")

    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """
        Add a notification handler function.

        Args:
            handler: Function that takes an Alert object
        """
        self.notification_handlers.append(handler)
        logger.info("Added notification handler")

    def check_alerts(self, metrics: Dict):
        """
        Check all alert rules against current metrics.

        Args:
            metrics: Dictionary of current system metrics
        """
        current_time = time.time()

        for rule in self.alert_rules:
            try:
                # Check cooldown
                if current_time - rule['last_triggered'] < rule['cooldown']:
                    continue

                # Check condition
                if rule['condition'](metrics):
                    self._trigger_alert(rule, metrics)
                    rule['last_triggered'] = current_time

            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {e}")

    def _trigger_alert(self, rule: Dict, metrics: Dict):
        """Trigger an alert for the given rule."""
        alert_id = f"{rule['name']}_{int(time.time())}"

        alert = Alert(
            id=alert_id,
            type=rule['type'],
            severity=rule['severity'],
            title=f"{rule['severity'].value.upper()}: {rule['name']}",
            message=rule['message'],
            timestamp=time.time(),
            metadata={
                'rule_name': rule['name'],
                'metrics': metrics,
                'cooldown': rule['cooldown']
            }
        )

        # Add to active alerts
        self.alerts.append(alert)

        # Record in Prometheus metrics
        record_alert(rule['type'].value, rule['severity'].value)

        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]

        # Send notifications
        self._send_notifications(alert)

        logger.warning(f"Alert triggered: {alert.title} - {alert.message}")

    def _send_notifications(self, alert: Alert):
        """Send alert to all notification handlers."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")

    def resolve_alert(self, alert_id: str, resolution_note: str = ""):
        """
        Resolve an active alert.

        Args:
            alert_id: ID of the alert to resolve
            resolution_note: Optional note about the resolution
        """
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = time.time()
                alert.metadata['resolution_note'] = resolution_note
                logger.info(f"Alert resolved: {alert_id}")
                break

    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active (unresolved) alerts."""
        return [alert for alert in self.alerts if not alert.resolved]

    def get_alert_history(self, limit: int = 50) -> List[Alert]:
        """Get recent alert history."""
        return self.alert_history[-limit:]

    def get_alert_stats(self) -> Dict:
        """Get alert statistics."""
        total_alerts = len(self.alert_history)
        active_alerts = len(self.get_active_alerts())

        severity_counts = {}
        type_counts = {}

        for alert in self.alert_history:
            severity_counts[alert.severity.value] = severity_counts.get(alert.severity.value, 0) + 1
            type_counts[alert.type.value] = type_counts.get(alert.type.value, 0) + 1

        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'severity_counts': severity_counts,
            'type_counts': type_counts
        }

# Default notification handlers

def log_alert_handler(alert: Alert):
    """Log alert to standard logging."""
    logger.warning(
        f"ALERT [{alert.severity.value.upper()}]: {alert.title} - {alert.message}"
    )

def json_alert_handler(alert: Alert):
    """Log alert as JSON for structured logging."""
    alert_data = {
        'id': alert.id,
        'type': alert.type.value,
        'severity': alert.severity.value,
        'title': alert.title,
        'message': alert.message,
        'timestamp': alert.timestamp,
        'metadata': alert.metadata
    }
    logger.warning(f"ALERT_JSON: {json.dumps(alert_data)}")

# Convenience functions for common use cases

def create_email_alert_handler(smtp_config: Dict) -> Callable[[Alert], None]:
    """
    Create an email notification handler.

    Args:
        smtp_config: SMTP configuration dictionary

    Returns:
        Alert handler function
    """
    def email_handler(alert: Alert):
        try:
            import smtplib
            from email.mime.text import MIMEText

            msg = MIMEText(f"""
            Alert: {alert.title}

            {alert.message}

            Severity: {alert.severity.value}
            Type: {alert.type.value}
            Time: {time.ctime(alert.timestamp)}

            Metadata: {json.dumps(alert.metadata, indent=2)}
            """)

            msg['Subject'] = f"SentinelFS Alert: {alert.title}"
            msg['From'] = smtp_config.get('from_email', 'alerts@sentinelfs.ai')
            msg['To'] = smtp_config.get('to_email', 'admin@sentinelfs.ai')

            server = smtplib.SMTP(smtp_config['host'], smtp_config.get('port', 587))
            if smtp_config.get('tls', True):
                server.starttls()
            if 'username' in smtp_config:
                server.login(smtp_config['username'], smtp_config['password'])

            server.sendmail(msg['From'], msg['To'], msg.as_string())
            server.quit()

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    return email_handler

def create_slack_alert_handler(webhook_url: str) -> Callable[[Alert], None]:
    """
    Create a Slack notification handler.

    Args:
        webhook_url: Slack webhook URL

    Returns:
        Alert handler function
    """
    def slack_handler(alert: Alert):
        try:
            import requests

            color_map = {
                AlertSeverity.INFO.value: "good",
                AlertSeverity.WARNING.value: "warning",
                AlertSeverity.ERROR.value: "danger",
                AlertSeverity.CRITICAL.value: "danger"
            }

            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity.value, "warning"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Type", "value": alert.type.value, "short": True},
                        {"title": "Time", "value": time.ctime(alert.timestamp), "short": False}
                    ]
                }]
            }

            requests.post(webhook_url, json=payload, timeout=5)

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    return slack_handler