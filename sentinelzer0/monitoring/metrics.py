"""
Prometheus Metrics for SentinelFS AI

This module defines all Prometheus metrics used for monitoring the AI system.
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time
from typing import Optional

# Request Metrics
REQUEST_COUNT = Counter(
    'sentinelfs_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'sentinelfs_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# Inference Metrics
INFERENCE_LATENCY = Histogram(
    'sentinelfs_inference_duration_seconds',
    'AI model inference duration in seconds',
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
)

PREDICTION_COUNT = Counter(
    'sentinelfs_predictions_total',
    'Total number of predictions made',
    ['result']  # 'threat', 'benign', 'error'
)

# Model Performance Metrics
MODEL_ACCURACY = Gauge(
    'sentinelfs_model_accuracy',
    'Current model accuracy score (0.0-1.0)'
)

MODEL_DRIFT_SCORE = Gauge(
    'sentinelfs_model_drift_score',
    'Model drift detection score (0.0-1.0, higher = more drift)'
)

# System Health Metrics
ACTIVE_CONNECTIONS = Gauge(
    'sentinelfs_active_connections',
    'Number of active connections'
)

MEMORY_USAGE = Gauge(
    'sentinelfs_memory_usage_bytes',
    'Current memory usage in bytes'
)

GPU_MEMORY_USAGE = Gauge(
    'sentinelfs_gpu_memory_usage_bytes',
    'Current GPU memory usage in bytes',
    ['device']
)

# Alerting Metrics
ALERT_COUNT = Counter(
    'sentinelfs_alerts_total',
    'Total number of alerts triggered',
    ['type', 'severity']  # 'drift', 'latency', 'error', etc.
)

# Model Information
MODEL_INFO = Info(
    'sentinelfs_model_info',
    'Information about the current model'
)

def init_metrics(
    model_name: str = "sentinelfs_ai",
    model_version: str = "1.0.0",
    model_type: str = "hybrid_gru"
):
    """
    Initialize model information metrics.

    Args:
        model_name: Name of the model
        model_version: Version of the model
        model_type: Type of model architecture
    """
    MODEL_INFO.info({
        'name': model_name,
        'version': model_version,
        'type': model_type,
        'framework': 'pytorch'
    })

def record_inference_time(start_time: float, labels: Optional[dict] = None):
    """
    Record inference latency.

    Args:
        start_time: Time when inference started (time.time())
        labels: Optional labels for the metric
    """
    duration = time.time() - start_time
    INFERENCE_LATENCY.observe(duration)

def record_prediction(result: str):
    """
    Record a prediction result.

    Args:
        result: 'threat', 'benign', or 'error'
    """
    PREDICTION_COUNT.labels(result=result).inc()

def update_model_accuracy(accuracy: float):
    """
    Update the model accuracy gauge.

    Args:
        accuracy: Accuracy score between 0.0 and 1.0
    """
    MODEL_ACCURACY.set(accuracy)

def update_drift_score(score: float):
    """
    Update the model drift score.

    Args:
        score: Drift score between 0.0 and 1.0
    """
    MODEL_DRIFT_SCORE.set(score)

def record_alert(alert_type: str, severity: str = "warning"):
    """
    Record an alert.

    Args:
        alert_type: Type of alert ('drift', 'latency', 'error', etc.)
        severity: Severity level ('info', 'warning', 'error', 'critical')
    """
    ALERT_COUNT.labels(type=alert_type, severity=severity).inc()

def update_memory_usage():
    """Update memory usage metrics."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    MEMORY_USAGE.set(memory_info.rss)

def update_gpu_memory_usage():
    """Update GPU memory usage if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i)
                GPU_MEMORY_USAGE.labels(device=str(i)).set(memory_allocated)
    except ImportError:
        pass