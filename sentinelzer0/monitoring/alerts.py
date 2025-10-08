"""
Enhanced Alerting System for SentinelFS AI

This module provides comprehensive alerting with multiple notification channels
including email, Slack, Discord, webhooks, and PagerDuty integration.
"""

import logging
import time
import smtplib
import json
import requests
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
from concurrent.futures import ThreadPoolExecutor
import os

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
    AVAILABILITY = "availability"

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

class AlertHandler:
    """Base class for alert handlers."""

    def send_alert(self, alert: Alert) -> bool:
        """Send an alert. Returns True if successful."""
        raise NotImplementedError

    def format_alert(self, alert: Alert) -> str:
        """Format alert for the specific channel."""
        return f"[{alert.severity.value.upper()}] {alert.title}\n\n{alert.message}"

class EmailAlertHandler(AlertHandler):
    """Email alert handler with HTML formatting."""

    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str,
                 from_email: str, to_emails: List[str], use_tls: bool = True):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.use_tls = use_tls

    def send_alert(self, alert: Alert) -> bool:
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🚨 SentinelFS Alert: {alert.title}"
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)

            # Plain text version
            text_body = self.format_alert(alert)
            msg.attach(MIMEText(text_body, 'plain'))

            # HTML version
            html_body = self._create_html_alert(alert)
            msg.attach(MIMEText(html_body, 'html'))

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.sendmail(self.from_email, self.to_emails, msg.as_string())
            server.quit()

            logger.info(f"Email alert sent successfully to {len(self.to_emails)} recipients")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _create_html_alert(self, alert: Alert) -> str:
        """Create HTML formatted alert."""
        severity_colors = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.ERROR: "#fd7e14",
            AlertSeverity.CRITICAL: "#dc3545"
        }

        color = severity_colors.get(alert.severity, "#6c757d")

        # Format metadata
        metadata_html = ""
        if alert.metadata:
            metadata_html = "<h4>Details:</h4><ul>"
            for key, value in alert.metadata.items():
                metadata_html += f"<li><strong>{key}:</strong> {value}</li>"
            metadata_html += "</ul>"

        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="border: 2px solid {color}; border-radius: 8px; padding: 20px; margin: 20px 0;">
                <h2 style="color: {color}; margin-top: 0;">
                    🚨 SentinelFS Alert
                </h2>
                <h3 style="margin-bottom: 10px;">{alert.title}</h3>
                <p style="font-size: 16px; line-height: 1.5;">{alert.message}</p>
                {metadata_html}
                <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
                <p style="color: #666; font-size: 12px;">
                    Alert ID: {alert.id}<br>
                    Severity: {alert.severity.value.upper()}<br>
                    Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}<br>
                    Type: {alert.type.value}
                </p>
            </div>
        </body>
        </html>
        """
        return html

class SlackAlertHandler(AlertHandler):
    """Slack alert handler with rich formatting."""

    def __init__(self, webhook_url: str, channel: str = None, username: str = "SentinelFS Alert"):
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username

    def send_alert(self, alert: Alert) -> bool:
        try:
            # Create Slack message
            color = self._get_slack_color(alert.severity)
            emoji = self._get_severity_emoji(alert.severity)

            message = {
                "username": self.username,
                "icon_emoji": emoji,
                "attachments": [{
                    "color": color,
                    "title": f"{emoji} {alert.title}",
                    "text": alert.message,
                    "fields": self._create_slack_fields(alert),
                    "footer": "SentinelFS AI Monitoring",
                    "ts": alert.timestamp
                }]
            }

            if self.channel:
                message["channel"] = self.channel

            # Send to Slack
            response = requests.post(self.webhook_url, json=message, timeout=10)
            response.raise_for_status()

            logger.info("Slack alert sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def _get_slack_color(self, severity: AlertSeverity) -> str:
        """Get Slack color for severity."""
        colors = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "danger",
            AlertSeverity.CRITICAL: "#dc3545"
        }
        return colors.get(severity, "#6c757d")

    def _get_severity_emoji(self, severity: AlertSeverity) -> str:
        """Get emoji for severity."""
        emojis = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }
        return emojis.get(severity, "📢")

    def _create_slack_fields(self, alert: Alert) -> List[Dict]:
        """Create Slack attachment fields."""
        fields = [
            {
                "title": "Severity",
                "value": alert.severity.value.upper(),
                "short": True
            },
            {
                "title": "Type",
                "value": alert.type.value.replace('_', ' ').title(),
                "short": True
            }
        ]

        # Add metadata fields
        if alert.metadata:
            for key, value in list(alert.metadata.items())[:3]:  # Limit to 3 fields
                fields.append({
                    "title": key.replace('_', ' ').title(),
                    "value": str(value),
                    "short": True
                })

        return fields

class DiscordAlertHandler(AlertHandler):
    """Discord alert handler."""

    def __init__(self, webhook_url: str, username: str = "SentinelFS Alert"):
        self.webhook_url = webhook_url
        self.username = username

    def send_alert(self, alert: Alert) -> bool:
        try:
            color = self._get_discord_color(alert.severity)
            emoji = self._get_severity_emoji(alert.severity)

            embed = {
                "title": f"{emoji} {alert.title}",
                "description": alert.message,
                "color": color,
                "fields": [
                    {
                        "name": "Severity",
                        "value": alert.severity.value.upper(),
                        "inline": True
                    },
                    {
                        "name": "Type",
                        "value": alert.type.value.replace('_', ' ').title(),
                        "inline": True
                    },
                    {
                        "name": "Time",
                        "value": time.ctime(alert.timestamp),
                        "inline": False
                    },
                    {
                        "name": "Alert ID",
                        "value": alert.id,
                        "inline": False
                    }
                ],
                "footer": {
                    "text": "SentinelFS AI Monitoring"
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(alert.timestamp))
            }

            # Add metadata
            if alert.metadata:
                metadata_text = "\n".join([f"• **{k}:** {v}" for k, v in alert.metadata.items() if k != 'metrics'])
                if metadata_text:
                    embed["fields"].append({
                        "name": "Details",
                        "value": metadata_text,
                        "inline": False
                    })

            message = {
                "username": self.username,
                "embeds": [embed]
            }

            response = requests.post(self.webhook_url, json=message, timeout=10)
            response.raise_for_status()

            logger.info("Discord alert sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False

    def _get_discord_color(self, severity: AlertSeverity) -> int:
        """Get Discord color for severity."""
        colors = {
            AlertSeverity.INFO: 0x17a2b8,
            AlertSeverity.WARNING: 0xffc107,
            AlertSeverity.ERROR: 0xfd7e14,
            AlertSeverity.CRITICAL: 0xdc3545
        }
        return colors.get(severity, 0x6c757d)

    def _get_severity_emoji(self, severity: AlertSeverity) -> str:
        """Get emoji for severity."""
        emojis = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }
        return emojis.get(severity, "📢")

class WebhookAlertHandler(AlertHandler):
    """Generic webhook alert handler."""

    def __init__(self, webhook_url: str, headers: Dict[str, str] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}

    def send_alert(self, alert: Alert) -> bool:
        try:
            payload = {
                "alert_id": alert.id,
                "severity": alert.severity.value,
                "type": alert.type.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "metadata": alert.metadata,
                "resolved": alert.resolved
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()

            logger.info("Webhook alert sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False

class PagerDutyAlertHandler(AlertHandler):
    """PagerDuty alert handler."""

    def __init__(self, routing_key: str, source: str = "sentinelfs-ai"):
        self.routing_key = routing_key
        self.source = source
        self.api_url = "https://events.pagerduty.com/v2/enqueue"

    def send_alert(self, alert: Alert) -> bool:
        try:
            severity = self._map_severity(alert.severity)

            payload = {
                "routing_key": self.routing_key,
                "event_action": "trigger" if not alert.resolved else "resolve",
                "dedup_key": alert.id,
                "payload": {
                    "summary": alert.title,
                    "source": self.source,
                    "severity": severity,
                    "component": "sentinelfs-ai",
                    "group": alert.type.value,
                    "class": alert.type.value,
                    "custom_details": alert.metadata,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(alert.timestamp))
                }
            }

            response = requests.post(self.api_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info(f"PagerDuty alert sent successfully (action: {payload['event_action']})")
            return True

        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False

    def _map_severity(self, severity: AlertSeverity) -> str:
        """Map SentinelFS severity to PagerDuty severity."""
        mapping = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical"
        }
        return mapping.get(severity, "info")

class AlertManager:
    """
    Enhanced alert manager with multiple notification channels.
    """

    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.handlers: List[AlertHandler] = []
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="alert")
        self.alert_history: List[Alert] = []
        self.max_history = 1000

    def add_handler(self, handler: AlertHandler):
        """Add an alert handler."""
        self.handlers.append(handler)
        logger.info(f"Added alert handler: {type(handler).__name__}")

    def remove_handler(self, handler: AlertHandler):
        """Remove an alert handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)
            logger.info(f"Removed alert handler: {type(handler).__name__}")

    def trigger_alert(self, alert_type: AlertType, severity: AlertSeverity,
                     title: str, message: str, metadata: Dict = None) -> str:
        """
        Trigger a new alert.

        Returns the alert ID.
        """
        alert_id = f"{alert_type.value}_{int(time.time() * 1000)}"

        alert = Alert(
            id=alert_id,
            type=alert_type,
            severity=severity,
            title=title,
            message=message,
            timestamp=time.time(),
            metadata=metadata or {}
        )

        self.alerts[alert_id] = alert
        self._add_to_history(alert)

        # Record metric
        record_alert(alert_type.value, severity.value)

        # Send notifications asynchronously
        self.executor.submit(self._send_notifications, alert)

        logger.warning(f"Alert triggered: {alert_id} - {title}")
        return alert_id

    def resolve_alert(self, alert_id: str, resolution_message: str = None):
        """Resolve an existing alert."""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = time.time()

            if resolution_message:
                alert.metadata["resolution"] = resolution_message

            # Send resolution notifications
            self.executor.submit(self._send_notifications, alert)

            logger.info(f"Alert resolved: {alert_id}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts.values() if not alert.resolved]

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get recent alert history."""
        return self.alert_history[-limit:]

    def _send_notifications(self, alert: Alert):
        """Send alert to all handlers."""
        if not self.handlers:
            logger.warning("No alert handlers configured")
            return

        successful_sends = 0
        for handler in self.handlers:
            try:
                if handler.send_alert(alert):
                    successful_sends += 1
                else:
                    logger.error(f"Handler {type(handler).__name__} failed to send alert")
            except Exception as e:
                logger.error(f"Error sending alert via {type(handler).__name__}: {e}")

        logger.info(f"Alert sent to {successful_sends}/{len(self.handlers)} handlers")

    def _add_to_history(self, alert: Alert):
        """Add alert to history, maintaining max size."""
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)

    def cleanup_old_alerts(self, max_age_hours: int = 24):
        """Clean up old resolved alerts."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        to_remove = []

        for alert_id, alert in self.alerts.items():
            if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time:
                to_remove.append(alert_id)

        for alert_id in to_remove:
            del self.alerts[alert_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old alerts")

# Global alert manager instance
alert_manager = AlertManager()

def trigger_drift_alert(drift_score: float, threshold: float):
    """Trigger a model drift alert."""
    title = f"Model Drift Detected: {drift_score:.3f}"
    message = f"Model drift score ({drift_score:.3f}) has exceeded threshold ({threshold:.3f})"
    metadata = {
        "drift_score": drift_score,
        "threshold": threshold,
        "timestamp": time.time()
    }

    severity = AlertSeverity.CRITICAL if drift_score > threshold * 1.5 else AlertSeverity.WARNING
    alert_manager.trigger_alert(AlertType.DRIFT, severity, title, message, metadata)

def trigger_latency_alert(p95_latency: float, threshold: float):
    """Trigger a latency alert."""
    title = f"High Latency: {p95_latency:.1f}ms"
    message = f"P95 response time ({p95_latency:.1f}ms) has exceeded threshold ({threshold:.1f}ms)"
    metadata = {
        "p95_latency_ms": p95_latency,
        "threshold_ms": threshold,
        "timestamp": time.time()
    }

    severity = AlertSeverity.CRITICAL if p95_latency > threshold * 2 else AlertSeverity.WARNING
    alert_manager.trigger_alert(AlertType.LATENCY, severity, title, message, metadata)

def trigger_error_rate_alert(error_rate: float, threshold: float):
    """Trigger an error rate alert."""
    title = f"High Error Rate: {error_rate:.1f}%"
    message = f"Error rate ({error_rate:.1f}%) has exceeded threshold ({threshold:.1f}%)"
    metadata = {
        "error_rate_percent": error_rate,
        "threshold_percent": threshold,
        "timestamp": time.time()
    }

    severity = AlertSeverity.CRITICAL if error_rate > threshold * 2 else AlertSeverity.ERROR
    alert_manager.trigger_alert(AlertType.ERROR_RATE, severity, title, message, metadata)

def trigger_memory_alert(memory_usage: float, threshold: float):
    """Trigger a memory usage alert."""
    title = f"High Memory Usage: {memory_usage:.1f}GB"
    message = f"Memory usage ({memory_usage:.1f}GB) has exceeded threshold ({threshold:.1f}GB)"
    metadata = {
        "memory_usage_gb": memory_usage,
        "threshold_gb": threshold,
        "timestamp": time.time()
    }

    severity = AlertSeverity.CRITICAL if memory_usage > threshold * 1.2 else AlertSeverity.WARNING
    alert_manager.trigger_alert(AlertType.MEMORY, severity, title, message, metadata)

def trigger_availability_alert(downtime_minutes: float, threshold: float):
    """Trigger an availability alert."""
    title = f"Service Unavailable: {downtime_minutes:.1f}min downtime"
    message = f"Service has been unavailable for {downtime_minutes:.1f} minutes (threshold: {threshold:.1f}min)"
    metadata = {
        "downtime_minutes": downtime_minutes,
        "threshold_minutes": threshold,
        "timestamp": time.time()
    }

    severity = AlertSeverity.CRITICAL
    alert_manager.trigger_alert(AlertType.AVAILABILITY, severity, title, message, metadata)

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
    Create an email notification handler with enhanced formatting.

    Args:
        smtp_config: SMTP configuration dictionary with keys:
            - host: SMTP server host
            - port: SMTP server port (default: 587)
            - username: SMTP username
            - password: SMTP password
            - from_email: From email address (default: alerts@sentinelfs.ai)
            - to_email: To email address (default: admin@sentinelfs.ai)
            - tls: Use TLS (default: True)
            - subject_prefix: Prefix for email subjects (default: "SentinelFS Alert")

    Returns:
        Alert handler function
    """
    def email_handler(alert: Alert):
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # Create message
            msg = MIMEMultipart('alternative')
            subject_prefix = smtp_config.get('subject_prefix', 'SentinelFS Alert')
            msg['Subject'] = f"{subject_prefix}: {alert.title}"
            msg['From'] = smtp_config.get('from_email', 'alerts@sentinelfs.ai')
            msg['To'] = smtp_config.get('to_email', 'admin@sentinelfs.ai')

            # Create HTML content
            html_content = f"""
            <html>
            <body>
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <h2 style="color: #d32f2f;">🚨 {alert.title}</h2>

                    <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0;">
                        <p style="margin: 0; font-size: 16px;">{alert.message}</p>
                    </div>

                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Severity:</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{alert.severity.value.upper()}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Type:</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{alert.type.value}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Time:</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{time.ctime(alert.timestamp)}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Alert ID:</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{alert.id}</td>
                        </tr>
                    </table>

                    <h3>Metadata:</h3>
                    <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto;">
{json.dumps(alert.metadata, indent=2)}
                    </pre>

                    <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
                    <p style="color: #666; font-size: 12px;">
                        This alert was generated by SentinelFS AI Monitoring System.<br>
                        Alert ID: {alert.id}
                    </p>
                </div>
            </body>
            </html>
            """

            # Create plain text content
            text_content = f"""
Alert: {alert.title}

{alert.message}

Severity: {alert.severity.value}
Type: {alert.type.value}
Time: {time.ctime(alert.timestamp)}
Alert ID: {alert.id}

Metadata:
{json.dumps(alert.metadata, indent=2)}

---
This alert was generated by SentinelFS AI Monitoring System.
Alert ID: {alert.id}
            """

            # Attach parts
            msg.attach(MIMEText(text_content, 'plain'))
            msg.attach(MIMEText(html_content, 'html'))

            # Send email
            server = smtplib.SMTP(smtp_config['host'], smtp_config.get('port', 587))
            if smtp_config.get('tls', True):
                server.starttls()
            if 'username' in smtp_config:
                server.login(smtp_config['username'], smtp_config['password'])

            server.sendmail(msg['From'], msg['To'], msg.as_string())
            server.quit()

            logger.info(f"Email alert sent successfully: {alert.id}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    return email_handler

def create_slack_alert_handler(webhook_url: str, channel: str = None, username: str = "SentinelFS AI") -> Callable[[Alert], None]:
    """
    Create a Slack notification handler with enhanced formatting.

    Args:
        webhook_url: Slack webhook URL
        channel: Slack channel to post to (optional, can be overridden in webhook)
        username: Bot username (default: "SentinelFS AI")

    Returns:
        Alert handler function
    """
    def slack_handler(alert: Alert):
        try:
            import requests

            # Color mapping for severity
            color_map = {
                AlertSeverity.INFO.value: "good",
                AlertSeverity.WARNING.value: "warning",
                AlertSeverity.ERROR.value: "danger",
                AlertSeverity.CRITICAL.value: "#d32f2f"
            }

            # Icon mapping for severity
            icon_map = {
                AlertSeverity.INFO.value: "ℹ️",
                AlertSeverity.WARNING.value: "⚠️",
                AlertSeverity.ERROR.value: "❌",
                AlertSeverity.CRITICAL.value: "🚨"
            }

            # Create attachment
            attachment = {
                "color": color_map.get(alert.severity.value, "warning"),
                "title": f"{icon_map.get(alert.severity.value, '⚠️')} {alert.title}",
                "text": alert.message,
                "fields": [
                    {
                        "title": "Severity",
                        "value": alert.severity.value.upper(),
                        "short": True
                    },
                    {
                        "title": "Type",
                        "value": alert.type.value,
                        "short": True
                    },
                    {
                        "title": "Time",
                        "value": time.ctime(alert.timestamp),
                        "short": False
                    },
                    {
                        "title": "Alert ID",
                        "value": alert.id,
                        "short": False
                    }
                ],
                "footer": "SentinelFS AI Monitoring",
                "ts": alert.timestamp
            }

            # Add metadata if present
            if alert.metadata:
                metadata_text = "\n".join([f"• {k}: {v}" for k, v in alert.metadata.items() if k != 'metrics'])
                if metadata_text:
                    attachment["fields"].append({
                        "title": "Details",
                        "value": metadata_text,
                        "short": False
                    })

            # Create payload
            payload = {
                "username": username,
                "attachments": [attachment]
            }

            if channel:
                payload["channel"] = channel

            # Send to Slack
            response = requests.post(webhook_url, json=payload, timeout=10)

            if response.status_code == 200:
                logger.info(f"Slack alert sent successfully: {alert.id}")
            else:
                logger.error(f"Slack alert failed with status {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    return slack_handler


def create_discord_alert_handler(webhook_url: str, username: str = "SentinelFS AI") -> Callable[[Alert], None]:
    """
    Create a Discord notification handler.

    Args:
        webhook_url: Discord webhook URL
        username: Bot username (default: "SentinelFS AI")

    Returns:
        Alert handler function
    """
    def discord_handler(alert: Alert):
        try:
            import requests

            # Color mapping for Discord embed
            color_map = {
                AlertSeverity.INFO.value: 0x00ff00,      # Green
                AlertSeverity.WARNING.value: 0xffff00,   # Yellow
                AlertSeverity.ERROR.value: 0xffa500,     # Orange
                AlertSeverity.CRITICAL.value: 0xff0000   # Red
            }

            # Emoji mapping for severity
            emoji_map = {
                AlertSeverity.INFO.value: "ℹ️",
                AlertSeverity.WARNING.value: "⚠️",
                AlertSeverity.ERROR.value: "❌",
                AlertSeverity.CRITICAL.value: "🚨"
            }

            # Create embed
            embed = {
                "title": f"{emoji_map.get(alert.severity.value, '⚠️')} {alert.title}",
                "description": alert.message,
                "color": color_map.get(alert.severity.value, 0xffff00),
                "fields": [
                    {
                        "name": "Severity",
                        "value": alert.severity.value.upper(),
                        "inline": True
                    },
                    {
                        "name": "Type",
                        "value": alert.type.value,
                        "inline": True
                    },
                    {
                        "name": "Time",
                        "value": time.ctime(alert.timestamp),
                        "inline": False
                    },
                    {
                        "name": "Alert ID",
                        "value": alert.id,
                        "inline": False
                    }
                ],
                "footer": {
                    "text": "SentinelFS AI Monitoring"
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(alert.timestamp))
            }

            # Add metadata if present
            if alert.metadata:
                metadata_text = "\n".join([f"• **{k}:** {v}" for k, v in alert.metadata.items() if k != 'metrics'])
                if metadata_text:
                    embed["fields"].append({
                        "name": "Details",
                        "value": metadata_text,
                        "inline": False
                    })

            # Create payload
            payload = {
                "username": username,
                "embeds": [embed]
            }

            # Send to Discord
            response = requests.post(webhook_url, json=payload, timeout=10)

            if response.status_code == 204:
                logger.info(f"Discord alert sent successfully: {alert.id}")
            else:
                logger.error(f"Discord alert failed with status {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")

    return discord_handler


def create_webhook_alert_handler(webhook_url: str, headers: Dict = None) -> Callable[[Alert], None]:
    """
    Create a generic webhook notification handler.

    Args:
        webhook_url: Webhook URL to send alerts to
        headers: Optional headers to include in the request

    Returns:
        Alert handler function
    """
    def webhook_handler(alert: Alert):
        try:
            import requests

            # Prepare payload
            payload = {
                "alert_id": alert.id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "type": alert.type.value,
                "timestamp": alert.timestamp,
                "metadata": alert.metadata,
                "resolved": alert.resolved,
                "source": "sentinelfs_ai"
            }

            # Prepare headers
            request_headers = {"Content-Type": "application/json"}
            if headers:
                request_headers.update(headers)

            # Send webhook
            response = requests.post(
                webhook_url,
                json=payload,
                headers=request_headers,
                timeout=10
            )

            if response.status_code in [200, 201, 202, 204]:
                logger.info(f"Webhook alert sent successfully: {alert.id}")
            else:
                logger.error(f"Webhook alert failed with status {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    return webhook_handler


def create_pagerduty_alert_handler(api_key: str, integration_key: str) -> Callable[[Alert], None]:
    """
    Create a PagerDuty notification handler.

    Args:
        api_key: PagerDuty API key
        integration_key: PagerDuty integration key (routing key)

    Returns:
        Alert handler function
    """
    def pagerduty_handler(alert: Alert):
        try:
            import requests

            # Severity mapping for PagerDuty
            severity_map = {
                AlertSeverity.INFO.value: "info",
                AlertSeverity.WARNING.value: "warning",
                AlertSeverity.ERROR.value: "error",
                AlertSeverity.CRITICAL.value: "critical"
            }

            # Create event payload
            payload = {
                "routing_key": integration_key,
                "event_action": "trigger",
                "dedup_key": alert.id,
                "payload": {
                    "summary": alert.message,
                    "source": "sentinelfs_ai",
                    "severity": severity_map.get(alert.severity.value, "warning"),
                    "component": "ai_model",
                    "group": alert.type.value,
                    "class": alert.severity.value,
                    "custom_details": alert.metadata,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(alert.timestamp))
                }
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Token token={api_key}"
            }

            # Send to PagerDuty
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                headers=headers,
                timeout=10
            )

            if response.status_code == 202:
                logger.info(f"PagerDuty alert sent successfully: {alert.id}")
            else:
                logger.error(f"PagerDuty alert failed with status {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")

    return pagerduty_handler