"""
REST API schemas and data models for SentinelZer0.

This module defines Pydantic models for request/response validation
and API documentation.

Phase 1.2: REST API Framework
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    """File system event types."""
    CREATE = "CREATE"
    MODIFY = "MODIFY"
    DELETE = "DELETE"
    RENAME = "RENAME"
    CHMOD = "CHMOD"
    CHOWN = "CHOWN"


class FileSystemEventRequest(BaseModel):
    """Single file system event for threat detection."""
    
    event_type: EventType = Field(..., description="Type of file system operation")
    path: str = Field(..., description="Full file path", min_length=1)
    timestamp: float = Field(..., description="Unix timestamp of the event")
    size: Optional[int] = Field(None, description="File size in bytes", ge=0)
    is_directory: bool = Field(False, description="Whether the path is a directory")
    user: Optional[str] = Field(None, description="User who triggered the event")
    process: Optional[str] = Field(None, description="Process name that caused the event")
    extension: Optional[str] = Field(None, description="File extension (e.g., '.txt')")
    
    class Config:
        json_json_schema_extra = {
            "example": {
                "event_type": "MODIFY",
                "path": "/home/user/document.txt",
                "timestamp": 1696723200.0,
                "size": 2048,
                "is_directory": False,
                "user": "john",
                "process": "python3",
                "extension": ".txt"
            }
        }
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Ensure timestamp is reasonable."""
        if v < 0 or v > datetime.now().timestamp() + 3600:
            raise ValueError('Invalid timestamp')
        return v


class BatchPredictionRequest(BaseModel):
    """Batch of file system events for threat detection."""
    
    events: List[FileSystemEventRequest] = Field(
        ..., 
        description="List of file system events",
        min_items=1,
        max_items=1000
    )
    return_components: bool = Field(
        False,
        description="Whether to return individual component scores (DL, IF, heuristic)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "events": [
                    {
                        "event_type": "CREATE",
                        "path": "/tmp/test.txt",
                        "timestamp": 1696723200.0,
                        "size": 1024,
                        "is_directory": False,
                        "user": "user",
                        "process": "bash"
                    }
                ],
                "return_components": False
            }
        }


class ThreatPredictionResponse(BaseModel):
    """Response for a single threat prediction."""
    
    event_id: str = Field(..., description="Event identifier (typically the path)")
    timestamp: float = Field(..., description="Event timestamp")
    threat_score: float = Field(..., description="Threat probability (0-1)", ge=0, le=1)
    is_threat: bool = Field(..., description="Whether event is classified as a threat")
    confidence: float = Field(..., description="Prediction confidence (0-1)", ge=0, le=1)
    latency_ms: float = Field(..., description="Processing latency in milliseconds")
    components: Optional[Dict[str, float]] = Field(
        None,
        description="Component scores if requested"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "/tmp/suspicious.exe",
                "timestamp": 1696723200.0,
                "threat_score": 0.87,
                "is_threat": True,
                "confidence": 0.94,
                "latency_ms": 0.82,
                "components": {
                    "dl_score": 0.85,
                    "if_score": 0.92,
                    "heuristic_score": 0.88
                }
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response for batch threat detection."""
    
    predictions: List[ThreatPredictionResponse] = Field(
        ...,
        description="List of predictions for each event"
    )
    total_events: int = Field(..., description="Total number of events processed")
    threats_detected: int = Field(..., description="Number of threats detected")
    total_latency_ms: float = Field(..., description="Total processing time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "event_id": "/tmp/file1.txt",
                        "timestamp": 1696723200.0,
                        "threat_score": 0.12,
                        "is_threat": False,
                        "confidence": 0.76,
                        "latency_ms": 0.85
                    }
                ],
                "total_events": 1,
                "threats_detected": 0,
                "total_latency_ms": 0.85
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.2.0",
                "model_loaded": True,
                "gpu_available": True,
                "uptime_seconds": 3600.5
            }
        }


class MetricsResponse(BaseModel):
    """Performance metrics response."""
    
    total_predictions: int = Field(..., description="Total predictions made")
    threats_detected: int = Field(..., description="Total threats detected")
    threat_rate: float = Field(..., description="Percentage of threats detected")
    avg_latency_ms: float = Field(..., description="Average prediction latency")
    p50_latency_ms: float = Field(..., description="50th percentile latency")
    p95_latency_ms: float = Field(..., description="95th percentile latency")
    p99_latency_ms: float = Field(..., description="99th percentile latency")
    max_latency_ms: float = Field(..., description="Maximum latency observed")
    buffer_size: int = Field(..., description="Current streaming buffer size")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_predictions": 10000,
                "threats_detected": 150,
                "threat_rate": 1.5,
                "avg_latency_ms": 0.82,
                "p50_latency_ms": 0.75,
                "p95_latency_ms": 1.2,
                "p99_latency_ms": 2.1,
                "max_latency_ms": 5.3,
                "buffer_size": 64
            }
        }


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    input_features: int = Field(..., description="Number of input features")
    sequence_length: int = Field(..., description="Sequence length for inference")
    threshold: float = Field(..., description="Classification threshold")
    architecture: Dict[str, Any] = Field(..., description="Model architecture details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "HybridThreatDetector",
                "version": "3.1.0",
                "input_features": 30,
                "sequence_length": 64,
                "threshold": 0.5,
                "architecture": {
                    "type": "Hybrid (GRU + Isolation Forest + Heuristics)",
                    "hidden_size": 64,
                    "num_layers": 2,
                    "components": ["deep_learning", "isolation_forest", "heuristic_rules"]
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: float = Field(..., description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid request format",
                "detail": "Field 'timestamp' is required",
                "timestamp": 1696723200.0
            }
        }


class StreamConfigRequest(BaseModel):
    """Configuration for streaming engine."""
    
    sequence_length: Optional[int] = Field(64, description="Buffer size", ge=1, le=256)
    threshold: Optional[float] = Field(0.5, description="Threat threshold", ge=0, le=1)
    min_confidence: Optional[float] = Field(0.6, description="Minimum confidence", ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "sequence_length": 64,
                "threshold": 0.5,
                "min_confidence": 0.6
            }
        }


class StreamConfigResponse(BaseModel):
    """Response after updating stream configuration."""
    
    success: bool = Field(..., description="Whether update was successful")
    message: str = Field(..., description="Status message")
    config: StreamConfigRequest = Field(..., description="Current configuration")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Configuration updated successfully",
                "config": {
                    "sequence_length": 64,
                    "threshold": 0.5,
                    "min_confidence": 0.6
                }
            }
        }
