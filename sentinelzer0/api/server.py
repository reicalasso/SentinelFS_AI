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

# Logger
logger = logging.getLogger(__name__)

# Global state
class AppState:
    """Application state container."""
    def __init__(self):
        self.model: Optional[HybridThreatDetector] = None
        self.engine: Optional[StreamingInferenceEngine] = None
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
MODEL_PATH = os.getenv('SENTINELFS_MODEL_PATH', 'models/production/sentinelfs_fixed.pt')


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
    # Startup
    logger.info("Starting SentinelZer0 API...")
    
    try:
        # Load model
        logger.info("Loading threat detection model...")
        
        # Try to load pre-trained weights if available
        model_loaded = False
        model_path = MODEL_PATH
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                
                # Infer architecture from checkpoint
                if 'rnn.weight_ih_l0' in state_dict:
                    # Calculate hidden_size from weight shape
                    weight_shape = state_dict['rnn.weight_ih_l0'].shape[0]
                    # For GRU: weight_shape = 3 * hidden_size (for bidirectional)
                    hidden_size = weight_shape // 6  # 3 gates * 2 directions
                    
                    # Calculate num_layers
                    num_layers = 1
                    for key in state_dict.keys():
                        if 'rnn.weight_ih_l' in key:
                            layer_num = int(key.split('_l')[1][0])
                            num_layers = max(num_layers, layer_num + 1)
                    
                    logger.info(f"Detected architecture: hidden_size={hidden_size}, num_layers={num_layers}")
                    
                    # Create model with matching architecture
                    app_state.model = HybridThreatDetector(
                        input_size=30,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        use_gru=True
                    )
                    
                    app_state.model.load_state_dict(state_dict)
                    logger.info("✓ Loaded pre-trained model with matching architecture")
                    model_loaded = True
                else:
                    raise ValueError("Invalid checkpoint format")
            else:
                raise ValueError("No model_state_dict in checkpoint")
                
        except FileNotFoundError:
            logger.warning(f"Model file not found: {model_path}")
        except Exception as e:
            logger.warning(f"Could not load pre-trained weights: {e}")
        
        # Fall back to default model if loading failed
        if not model_loaded:
            logger.info("Using randomly initialized model (default architecture)")
            app_state.model = HybridThreatDetector(input_size=30)
        
        # Initialize streaming engine
        logger.info("Initializing streaming inference engine...")
        app_state.engine = StreamingInferenceEngine(
            model=app_state.model,
            sequence_length=64,
            threshold=0.5,
            device='auto'
        )
        
        logger.info(f"✓ API started on device: {app_state.engine.device}")
        logger.info(f"✓ Streaming engine ready with buffer size: {app_state.engine.sequence_length}")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down SentinelZer0 API...")
    if app_state.engine:
        app_state.engine.clear_buffer()
    logger.info("✓ Shutdown complete")


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


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"Response: {response.status_code} | "
        f"Time: {process_time:.2f}ms | "
        f"Path: {request.url.path}"
    )
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
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
        predictions = app_state.engine.process_batch(
            events,
            return_components=request.return_components
        )
        
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
        
        total_latency = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            total_events=len(request.events),
            threats_detected=threats_detected,
            total_latency_ms=total_latency
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
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
            logger.warning("Sequence length change requires engine restart")
        
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
        logger.error(f"Configuration update error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Configuration update failed: {str(e)}"
        )


@app.post(
    "/metrics/reset",
    tags=["Monitoring"],
    summary="Reset performance metrics",
    dependencies=[Depends(verify_api_key)]
)
async def reset_metrics():
    """
    Reset performance statistics.
    
    Clears all accumulated metrics and counters.
    
    Requires authentication.
    """
    if not app_state.engine:
        raise HTTPException(
            status_code=503,
            detail="Streaming engine not initialized"
        )
    
    app_state.engine.reset_stats()
    
    return {
        "success": True,
        "message": "Metrics reset successfully",
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "sentinelzer0.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
