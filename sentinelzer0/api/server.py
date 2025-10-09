"""
SentinelZer0 REST API Server

FastAPI application providing real-time threat detection endpoints.

Phase 1.2: REST API Framework
"""

import time
import logging
import os
from datetime import datetime
from typing import Optional, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Security, status, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch

from ..models.hybrid_detector import HybridThreatDetector
from ..inference.streaming_engine import StreamingInferenceEngine, ThreatPrediction
from .schemas import (
    FileSystemEventRequest,
    BatchPredictionRequest,
    ThreatPredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    ErrorResponse,
    StreamConfigRequest,
    StreamConfigResponse
)

# Import monitoring components
from ..monitoring.metrics import (
    init_metrics,
    record_inference_time,
    record_prediction,
    update_model_accuracy,
    REQUEST_COUNT,
    REQUEST_LATENCY
)
from ..monitoring.middleware import PrometheusMiddleware, MetricsEndpoint
from ..monitoring.drift_detector import ModelDriftDetector
from ..monitoring.alerts import AlertManager, log_alert_handler, json_alert_handler
from ..monitoring.logging_config import setup_logging, logger

# Logger
logger = logging.getLogger(__name__)

# Global state
class AppState:
    """Application state container."""
    def __init__(self):
        self.model: Optional[HybridThreatDetector] = None
        self.engine: Optional[StreamingInferenceEngine] = None
        self.drift_detector: Optional[ModelDriftDetector] = None
        self.alert_manager: Optional[AlertManager] = None
        self.start_time: float = time.time()
        self.api_version: str = "1.2.0"
        self.model_version: str = "3.1.0"

app_state = AppState()

# Configuration from environment variables
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Load API keys from environment variable or use default for development
def get_valid_api_keys() -> Set[str]:
    """Get valid API keys from environment or use default."""
    env_keys = os.getenv('SENTINELFS_API_KEYS', '')
    if env_keys:
        return set(key.strip() for key in env_keys.split(',') if key.strip())
    # Development default
    return {"sentinelfs-dev-key-2025"}

VALID_API_KEYS = get_valid_api_keys()
CORS_ORIGINS = os.getenv('SENTINELFS_CORS_ORIGINS', '*').split(',')
MODEL_PATH = os.getenv('SENTINELFS_MODEL_PATH', 'models/production/trained_model.pt')
COMPONENTS_DIR_CANDIDATES = [
    os.getenv('SENTINELFS_COMPONENTS_DIR', ''),
    'checkpoints/final',
    'models/production/final'
]
METRICS_PATH = os.getenv('SENTINELFS_METRICS_PATH', 'models/production/training_metrics.json')


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key authentication."""
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key"
        )
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Setup structured logging
    global logger
    logger = setup_logging(level="INFO", json_format=True)

    # Startup
    logger.logger.info("Starting SentinelZer0 API...")
    
    try:
        # Initialize monitoring
        logger.logger.info("Initializing monitoring components...")
        
        # Initialize Prometheus metrics
        init_metrics(
            model_name="sentinelzer0_ai",
            model_version=app_state.model_version,
            model_type="hybrid_gru_isolation_forest"
        )
        
        # Initialize drift detector
        app_state.drift_detector = ModelDriftDetector(
            window_size=1000,
            drift_threshold=0.05,
            alert_threshold=0.1
        )
        
        # Set baseline for drift detection (use some initial predictions)
        logger.logger.info("Setting baseline for drift detection...")
        try:
            # Generate baseline predictions using the model
            baseline_events = [
                {
                    "event_type": "CREATE",
                    "path": f"/tmp/test_file_{i}.txt",
                    "timestamp": time.time() + i,
                    "size": 1024 + (i * 100),
                    "is_directory": False
                }
                for i in range(100)  # Generate 100 baseline events
            ]
            
            baseline_predictions = app_state.engine.process_batch(baseline_events, return_components=False)
            baseline_scores = [pred.threat_score for pred in baseline_predictions]
            
            app_state.drift_detector.set_baseline(baseline_scores)
            logger.logger.info(f"✓ Set drift detection baseline with {len(baseline_scores)} samples")
            
        except Exception as e:
            logger.logger.warning(f"Could not set drift detection baseline: {e}")
            # Continue without baseline - drift detection will start learning after some predictions
        
        # Initialize alert manager
        app_state.alert_manager = AlertManager()
        app_state.alert_manager.add_notification_handler(log_alert_handler)
        app_state.alert_manager.add_notification_handler(json_alert_handler)
        
        logger.logger.info("✓ Monitoring components initialized")
        
        # Load model
        logger.logger.info("Loading threat detection model...")
        
        # Try to load pre-trained weights if available
        model_loaded = False
        model_path = MODEL_PATH
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                
                # Infer architecture from checkpoint
                if 'rnn.weight_ih_l0' in state_dict:
                    # Calculate hidden_size from weight shape
                    weight_shape = state_dict['rnn.weight_ih_l0'].shape[0]
                    # For GRU with bidirectional: weight_shape = 3 * hidden_size * 2
                    # For LSTM with bidirectional: weight_shape = 4 * hidden_size * 2
                    hidden_size = weight_shape // 6  # 3 gates * 2 directions for GRU
                    
                    # Check if LSTM (4 gates)
                    is_lstm = 'lstm' in str(checkpoint.get('model_config', {})).lower()
                    if is_lstm or weight_shape % 8 == 0:  # LSTM has 4 gates
                        hidden_size = weight_shape // 8  # 4 gates * 2 directions
                    
                    # Count layers
                    num_layers = 1
                    for key in state_dict.keys():
                        if 'rnn.weight_ih_l' in key and not 'reverse' in key:
                            layer_num = int(key.split('_l')[1][0])
                            num_layers = max(num_layers, layer_num + 1)
                    
                    # Try to get from checkpoint config if available
                    if 'model_config' in checkpoint:
                        input_size = checkpoint['model_config'].get('input_size', 33)  # Default to 33 features
                        hidden_size = checkpoint['model_config'].get('hidden_size', hidden_size)
                        num_layers = checkpoint['model_config'].get('num_layers', num_layers)
                        dropout = checkpoint['model_config'].get('dropout', 0.3)
                    else:
                        input_size = 33  # New feature count
                        dropout = 0.3
                    
                    logger.logger.info(f"Detected architecture: input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}")
                    
                    # Create model with matching architecture
                    app_state.model = HybridThreatDetector(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        use_gru=False  # LSTM by default
                    )
                    
                    app_state.model.load_state_dict(state_dict)
                    logger.logger.info("✓ Loaded pre-trained model with matching architecture")
                    model_loaded = True
                else:
                    raise ValueError("Invalid checkpoint format")
            else:
                raise ValueError("No model_state_dict in checkpoint")
                
        except FileNotFoundError:
            logger.logger.warning(f"Model file not found: {model_path}")
        except Exception as e:
            logger.logger.warning(f"Could not load pre-trained weights: {e}")
        
        # Fall back to default model if loading failed
        if not model_loaded:
            logger.logger.info("Using randomly initialized model (default architecture)")
            app_state.model = HybridThreatDetector(input_size=33)  # New feature count
        
        # Load non-PyTorch components if available (Isolation Forest, thresholds)
        for comp_dir in COMPONENTS_DIR_CANDIDATES:
            if comp_dir and os.path.isdir(comp_dir):
                try:
                    app_state.model.load_components(comp_dir)
                    logger.logger.info(f"✓ Loaded model components from {comp_dir}")
                    break
                except Exception as e:
                    logger.logger.warning(f"Could not load components from {comp_dir}: {e}")

        # Read decision threshold and sequence length from metrics if available
        sequence_length = 64
        threshold = 0.5
        try:
            if os.path.isfile(METRICS_PATH):
                import json
                with open(METRICS_PATH, 'r') as f:
                    metrics = json.load(f)
                threshold = float(metrics.get('decision_threshold', threshold))
                # Prefer training sequence length if present in checkpoint or metrics
                sequence_length = int(metrics.get('sequence_length', sequence_length))
                logger.logger.info(f"Using trained params -> threshold={threshold:.3f}, sequence_length={sequence_length}")
        except Exception as e:
            logger.logger.warning(f"Could not read training metrics: {e}")

        # Initialize streaming engine
        logger.logger.info("Initializing streaming inference engine...")
        app_state.engine = StreamingInferenceEngine(
            model=app_state.model,
            sequence_length=sequence_length,
            threshold=threshold,
            device='auto'
        )
        
        logger.logger.info(f"✓ API started on device: {app_state.engine.device}")
        logger.logger.info(f"✓ Streaming engine ready with buffer size: {app_state.engine.sequence_length}")
        
    except Exception as e:
        logger.logger.error(f"Failed to initialize API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.logger.info("Shutting down SentinelZer0 API...")
    if app_state.engine:
        app_state.engine.clear_buffer()
    logger.logger.info("✓ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="SentinelZer0 API",
    description="""
    Real-time threat detection API for file system monitoring.
    
    Features:
    - Real-time event stream processing
    - Sub-millisecond inference latency
    - GPU acceleration support
    - Batch prediction capabilities
    - Comprehensive performance metrics
    
    **Authentication**: All endpoints require an API key in the `X-API-Key` header.
    """,
    version="1.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics endpoint
MetricsEndpoint.add_to_app(app, endpoint="/prometheus/metrics")


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    # Log request
    logger.logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = (time.time() - start_time) * 1000
    logger.logger.info(
        f"Response: {response.status_code} | "
        f"Time: {process_time:.2f}ms | "
        f"Path: {request.url.path}"
    )
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            detail=str(exc),
            timestamp=time.time()
        ).dict()
    )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SentinelZer0 API",
        "version": app_state.api_version,
        "status": "operational",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check():
    """
    Check API health status.
    
    Returns service health, model status, and GPU availability.
    """
    uptime = time.time() - app_state.start_time
    
    return HealthResponse(
        status="healthy" if app_state.model and app_state.engine else "degraded",
        version=app_state.api_version,
        model_loaded=app_state.model is not None,
        gpu_available=torch.cuda.is_available(),
        uptime_seconds=uptime
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Monitoring"],
    summary="Get performance metrics",
    dependencies=[Depends(verify_api_key)]
)
async def get_metrics():
    """
    Get real-time performance metrics.
    
    Requires authentication.
    """
    if not app_state.engine:
        raise HTTPException(
            status_code=503,
            detail="Streaming engine not initialized"
        )
    
    stats = app_state.engine.get_performance_stats()
    
    return MetricsResponse(
        total_predictions=stats['total_predictions'],
        threats_detected=int(stats['total_predictions'] * stats['threat_rate']),
        threat_rate=stats['threat_rate'] * 100,
        avg_latency_ms=stats['avg_latency_ms'],
        p50_latency_ms=stats['p50_latency_ms'],
        p95_latency_ms=stats['p95_latency_ms'],
        p99_latency_ms=stats['p99_latency_ms'],
        max_latency_ms=stats['max_latency_ms'],
        buffer_size=app_state.engine.buffer_size
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Get model information",
    dependencies=[Depends(verify_api_key)]
)
async def get_model_info():
    """
    Get detailed model information.
    
    Requires authentication.
    """
    if not app_state.model or not app_state.engine:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return ModelInfoResponse(
        model_name="HybridThreatDetector",
        version=app_state.model_version,
        input_features=app_state.model.input_size,
        sequence_length=app_state.engine.sequence_length,
        threshold=app_state.engine.threshold,
        architecture={
            "type": "Hybrid (GRU + Isolation Forest + Heuristics)",
            "hidden_size": app_state.model.hidden_size,
            "num_layers": app_state.model.num_layers,
            "use_gru": app_state.model.use_gru,
            "components": ["deep_learning", "isolation_forest", "heuristic_rules"],
            "device": str(app_state.engine.device)
        }
    )


@app.post(
    "/predict",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict threats from file system events",
    dependencies=[Depends(verify_api_key)]
)
async def predict_threats(request: BatchPredictionRequest):
    """
    Analyze file system events for potential threats.
    
    Processes a batch of file system events and returns threat predictions
    with confidence scores and latency metrics.
    
    Requires authentication.
    """
    if not app_state.engine:
        raise HTTPException(
            status_code=503,
            detail="Streaming engine not initialized"
        )
    
    start_time = time.time()
    
    try:
        # Convert request events to dict format
        events = [event.dict() for event in request.events]
        
        # Process events through streaming engine
        inference_start = time.time()
        predictions = app_state.engine.process_batch(
            events,
            return_components=request.return_components
        )
        inference_time = time.time() - inference_start
        
        # Record inference metrics
        record_inference_time(inference_start)
        
        # Convert to response format
        prediction_responses = []
        threats_detected = 0
        
        for pred in predictions:
            response = ThreatPredictionResponse(
                event_id=pred.event_id,
                timestamp=pred.timestamp,
                threat_score=pred.threat_score,
                is_threat=pred.is_threat,
                confidence=pred.confidence,
                latency_ms=pred.latency_ms,
                components=pred.components if request.return_components else None
            )
            prediction_responses.append(response)
            
            if pred.is_threat:
                threats_detected += 1
            
            # Record prediction result for monitoring
            result = "threat" if pred.is_threat else "benign"
            record_prediction(result)
            
            # Add to drift detector
            if app_state.drift_detector:
                app_state.drift_detector.add_prediction(pred.threat_score)
        
        total_latency = (time.time() - start_time) * 1000
        
        # Check for alerts
        if app_state.alert_manager:
            metrics = {
                'avg_latency': total_latency / len(request.events),
                'error_rate': 0.0,  # No errors in this request
                'drift_score': app_state.drift_detector.get_drift_status().get('drift_score', 0.0) if app_state.drift_detector else 0.0
            }
            app_state.alert_manager.check_alerts(metrics)
        
        # Log prediction performance
        logger.log_prediction(
            event_count=len(request.events),
            threat_count=threats_detected,
            latency_ms=total_latency,
            drift_score=app_state.drift_detector.get_drift_status().get('drift_score', 0.0) if app_state.drift_detector else 0.0
        )
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            total_events=len(request.events),
            threats_detected=threats_detected,
            total_latency_ms=total_latency
        )
        
    except Exception as e:
        logger.logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/config/stream",
    response_model=StreamConfigResponse,
    tags=["Configuration"],
    summary="Update streaming engine configuration",
    dependencies=[Depends(verify_api_key)]
)
async def update_stream_config(config: StreamConfigRequest):
    """
    Update streaming engine configuration.
    
    Allows dynamic adjustment of sequence length, threshold, and confidence settings.
    
    Requires authentication.
    """
    if not app_state.engine:
        raise HTTPException(
            status_code=503,
            detail="Streaming engine not initialized"
        )
    
    try:
        # Update configuration
        if config.threshold is not None:
            app_state.engine.threshold = config.threshold
        
        if config.min_confidence is not None:
            app_state.engine.min_confidence = config.min_confidence
        
        # Note: sequence_length cannot be changed without reinitializing engine
        if config.sequence_length != app_state.engine.sequence_length:
            logger.logger.warning("Sequence length change requires engine restart")
        
        return StreamConfigResponse(
            success=True,
            message="Configuration updated successfully",
            config=StreamConfigRequest(
                sequence_length=app_state.engine.sequence_length,
                threshold=app_state.engine.threshold,
                min_confidence=app_state.engine.min_confidence
            )
        )
        
    except Exception as e:
        logger.logger.error(f"Configuration update error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Configuration update failed: {str(e)}"
        )


@app.get(
    "/monitoring/drift",
    tags=["Monitoring"],
    summary="Get drift detection status",
    dependencies=[Depends(verify_api_key)]
)
async def get_drift_status():
    """
    Get current model drift detection status.
    
    Returns information about drift detection, including:
    - Whether drift has been detected
    - Current drift score
    - Detection method used
    - Baseline status
    
    Requires authentication.
    """
    if not app_state.drift_detector:
        raise HTTPException(
            status_code=503,
            detail="Drift detector not initialized"
        )
    
    status = app_state.drift_detector.get_drift_status()
    
    return {
        "drift_detected": status.get("has_drift", False),
        "drift_score": status.get("drift_score", 0.0),
        "confidence": status.get("confidence", 0.0),
        "method": status.get("method", "unknown"),
        "baseline_set": status.get("baseline_set", False),
        "samples_collected": status.get("samples_collected", 0),
        "last_check": status.get("last_check", 0),
        "history_size": status.get("history_size", 0)
    }


@app.post(
    "/monitoring/drift/reset",
    tags=["Monitoring"],
    summary="Reset drift detection baseline",
    dependencies=[Depends(verify_api_key)]
)
async def reset_drift_baseline():
    """
    Reset the drift detection baseline with current prediction data.
    
    This will use the most recent predictions as the new baseline
    for future drift detection.
    
    Requires authentication.
    """
    if not app_state.drift_detector:
        raise HTTPException(
            status_code=503,
            detail="Drift detector not initialized"
        )
    
    try:
        app_state.drift_detector.reset_baseline()
        
        return {
            "success": True,
            "message": "Drift detection baseline reset successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.logger.error(f"Failed to reset drift baseline: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset baseline: {str(e)}"
        )


@app.get(
    "/monitoring/alerts",
    tags=["Monitoring"],
    summary="Get active alerts",
    dependencies=[Depends(verify_api_key)]
)
async def get_active_alerts():
    """
    Get all currently active (unresolved) alerts.
    
    Returns a list of active alerts with their details.
    
    Requires authentication.
    """
    if not app_state.alert_manager:
        raise HTTPException(
            status_code=503,
            detail="Alert manager not initialized"
        )
    
    alerts = app_state.alert_manager.get_active_alerts()
    
    return {
        "alerts": [
            {
                "id": alert.id,
                "type": alert.type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "metadata": alert.metadata
            }
            for alert in alerts
        ],
        "total_active": len(alerts)
    }


@app.get(
    "/monitoring/alerts/history",
    tags=["Monitoring"],
    summary="Get alert history",
    dependencies=[Depends(verify_api_key)]
)
async def get_alert_history(limit: int = 50):
    """
    Get recent alert history.
    
    Parameters:
    - limit: Maximum number of alerts to return (default: 50)
    
    Requires authentication.
    """
    if not app_state.alert_manager:
        raise HTTPException(
            status_code=503,
            detail="Alert manager not initialized"
        )
    
    alerts = app_state.alert_manager.get_alert_history(limit)
    
    return {
        "alerts": [
            {
                "id": alert.id,
                "type": alert.type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "resolved": alert.resolved,
                "resolved_at": alert.resolved_at,
                "metadata": alert.metadata
            }
            for alert in alerts
        ],
        "total_returned": len(alerts)
    }


@app.post(
    "/monitoring/alerts/{alert_id}/resolve",
    tags=["Monitoring"],
    summary="Resolve an alert",
    dependencies=[Depends(verify_api_key)]
)
async def resolve_alert(alert_id: str, note: str = ""):
    """
    Resolve a specific alert.
    
    Parameters:
    - alert_id: ID of the alert to resolve
    - note: Optional resolution note
    
    Requires authentication.
    """
    if not app_state.alert_manager:
        raise HTTPException(
            status_code=503,
            detail="Alert manager not initialized"
        )
    
    try:
        app_state.alert_manager.resolve_alert(alert_id, note)
        
        return {
            "success": True,
            "message": f"Alert {alert_id} resolved successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.logger.error(f"Failed to resolve alert {alert_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resolve alert: {str(e)}"
        )


@app.get(
    "/monitoring/alerts/stats",
    tags=["Monitoring"],
    summary="Get alert statistics",
    dependencies=[Depends(verify_api_key)]
)
async def get_alert_stats():
    """
    Get alert statistics and summary.
    
    Returns counts by severity and type, plus other statistics.
    
    Requires authentication.
    """
    if not app_state.alert_manager:
        raise HTTPException(
            status_code=503,
            detail="Alert manager not initialized"
        )
    
    stats = app_state.alert_manager.get_alert_stats()
    
    return stats


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "sentinelzer0.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
