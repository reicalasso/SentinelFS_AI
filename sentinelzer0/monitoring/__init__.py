"""
Production Monitoring Module for SentinelFS AI

This module provides comprehensive monitoring capabilities including:
- Prometheus metrics exporter
- Model drift detection
- Performance logging
- Alerting system
- Grafana dashboard integration
"""

from .metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    INFERENCE_LATENCY,
    PREDICTION_COUNT,
    MODEL_ACCURACY,
    MODEL_DRIFT_SCORE,
    ALERT_COUNT,
    init_metrics
)

from .middleware import PrometheusMiddleware
from .drift_detector import ModelDriftDetector
from .alerts import AlertManager

__all__ = [
    'REQUEST_COUNT',
    'REQUEST_LATENCY',
    'INFERENCE_LATENCY',
    'PREDICTION_COUNT',
    'MODEL_ACCURACY',
    'MODEL_DRIFT_SCORE',
    'ALERT_COUNT',
    'init_metrics',
    'PrometheusMiddleware',
    'ModelDriftDetector',
    'AlertManager'
]