"""
Enhanced Logging Configuration for SentinelFS AI

This module provides structured logging with performance metrics,
JSON formatting, and configurable log levels.
"""

import logging
import logging.config
import json
import time
from typing import Dict, Any, Optional
import sys
from pathlib import Path

class StructuredLogger:
    """
    Structured logger with JSON formatting and performance tracking.
    """

    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create console handler with JSON formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(console_handler)

        # Create file handler for performance logs
        file_handler = logging.FileHandler("sentinelfs_performance.log")
        file_handler.setFormatter(JSONFormatter())
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)

        # Prevent duplicate logs from parent loggers
        self.logger.propagate = False

    def log_performance(self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """Log performance metrics."""
        log_data = {
            "operation": operation,
            "duration_ms": duration,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.logger.info("PERFORMANCE", extra={"structured_data": log_data})

    def log_request(self, method: str, path: str, status: int, duration: float, user_agent: str = ""):
        """Log HTTP request details."""
        log_data = {
            "method": method,
            "path": path,
            "status": status,
            "duration_ms": duration,
            "user_agent": user_agent,
            "timestamp": time.time()
        }
        self.logger.info("REQUEST", extra={"structured_data": log_data})

    def log_prediction(self, event_count: int, threat_count: int, latency_ms: float, drift_score: float = 0.0):
        """Log prediction batch details."""
        log_data = {
            "event_count": event_count,
            "threat_count": threat_count,
            "latency_ms": latency_ms,
            "drift_score": drift_score,
            "timestamp": time.time()
        }
        self.logger.info("PREDICTION", extra={"structured_data": log_data})

    def log_alert(self, alert_type: str, severity: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log alert events."""
        log_data = {
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        self.logger.warning("ALERT", extra={"structured_data": log_data})

    def log_error(self, error_type: str, message: str, traceback: str = "", metadata: Optional[Dict[str, Any]] = None):
        """Log error events with full context."""
        log_data = {
            "error_type": error_type,
            "message": message,
            "traceback": traceback,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        self.logger.error("ERROR", extra={"structured_data": log_data})

class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    """

    def format(self, record):
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }

        # Add structured data if present
        if hasattr(record, 'structured_data'):
            log_entry.update(record.structured_data)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)

def setup_logging(
    level: str = "INFO",
    log_file: str = "sentinelfs.log",
    performance_file: str = "sentinelfs_performance.log",
    json_format: bool = True
) -> StructuredLogger:
    """
    Setup comprehensive logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Main log file path
        performance_file: Performance log file path
        json_format: Whether to use JSON formatting

    Returns:
        Configured StructuredLogger instance
    """
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Setup basic logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Return structured logger instance
    return StructuredLogger('sentinelfs_ai', level)

    # Return structured logger instance
    return StructuredLogger('sentinelfs_ai', level)

# Global logger instance
logger = StructuredLogger('sentinelfs_ai')