"""
Real-time stream processing for SentinelZer0.

This module implements sliding window buffers and streaming inference
to enable continuous file system event analysis with <25ms latency.

Phase 1.1: Real-Time Stream Processing
"""

import time
from collections import deque
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass
from threading import Lock

from ..data.real_feature_extractor import RealFeatureExtractor
from ..models.hybrid_detector import HybridThreatDetector


@dataclass
class ThreatPrediction:
    """Result from streaming inference."""
    event_id: str
    timestamp: float
    threat_score: float
    is_threat: bool
    confidence: float
    latency_ms: float
    components: Optional[Dict[str, float]] = None


class StreamBuffer:
    """
    Sliding window buffer for real-time event streams.
    
    Maintains a fixed-size window of recent events and provides
    efficient access for streaming inference.
    
    Features:
    - Thread-safe operations
    - O(1) append and retrieval
    - Automatic old event eviction
    - Memory-efficient deque-based storage
    """
    
    def __init__(
        self, 
        max_size: int = 64,
        feature_extractor: Optional[RealFeatureExtractor] = None
    ):
        """
        Initialize stream buffer.
        
        Args:
            max_size: Maximum number of events to keep in buffer
            feature_extractor: Feature extractor for real-time processing
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.feature_cache = deque(maxlen=max_size)
        self.feature_extractor = feature_extractor or RealFeatureExtractor()
        self.lock = Lock()
        
        self._event_count = 0
        self._total_events = 0
    
    def add_event(self, event: Dict) -> bool:
        """
        Add new event to buffer.
        
        Args:
            event: File system event dictionary
            
        Returns:
            True if buffer is full and ready for inference
        """
        with self.lock:
            # Extract features immediately
            features = self.feature_extractor.extract_from_event(event)
            
            # Add to buffers
            self.buffer.append(event)
            self.feature_cache.append(features)
            
            self._event_count += 1
            self._total_events += 1
            
            return len(self.buffer) >= self.max_size
    
    def get_sequence(self, seq_length: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get current sequence from buffer for inference.
        
        Args:
            seq_length: Desired sequence length (uses buffer size if None)
            
        Returns:
            Feature array of shape (seq_length, num_features) or None if insufficient data
        """
        with self.lock:
            if seq_length is None:
                seq_length = self.max_size
            
            if len(self.feature_cache) < seq_length:
                return None
            
            # Get most recent seq_length features
            sequence = list(self.feature_cache)[-seq_length:]
            return np.array(sequence, dtype=np.float32)
    
    def get_events(self, count: Optional[int] = None) -> List[Dict]:
        """
        Get recent events from buffer.
        
        Args:
            count: Number of recent events to retrieve (all if None)
            
        Returns:
            List of event dictionaries
        """
        with self.lock:
            if count is None:
                return list(self.buffer)
            return list(self.buffer)[-count:]
    
    def clear(self):
        """Clear buffer and reset counters."""
        with self.lock:
            self.buffer.clear()
            self.feature_cache.clear()
            self._event_count = 0
    
    def is_ready(self, min_events: int = None) -> bool:
        """Check if buffer has enough events for inference."""
        min_events = min_events or self.max_size
        with self.lock:
            return len(self.buffer) >= min_events
    
    @property
    def size(self) -> int:
        """Current buffer size."""
        return len(self.buffer)
    
    @property
    def total_processed(self) -> int:
        """Total events processed since initialization."""
        return self._total_events
    
    def __len__(self) -> int:
        return self.size


class StreamingInferenceEngine:
    """
    Real-time inference engine for continuous event stream analysis.
    
    Processes file system events as they arrive using a sliding window
    approach with optimized feature extraction and model inference.
    
    Performance target: <25ms per event processing
    """
    
    def __init__(
        self,
        model: HybridThreatDetector,
        sequence_length: int = 64,
        threshold: float = 0.5,
        min_confidence: float = 0.6,
        device: str = 'auto'
    ):
        """
        Initialize streaming inference engine.
        
        Args:
            model: Trained HybridThreatDetector model
            sequence_length: Length of event sequences for inference
            threshold: Threat detection threshold
            min_confidence: Minimum confidence for threat alerts
            device: Device for inference ('cpu', 'cuda', or 'auto')
        """
        self.model = model
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.min_confidence = min_confidence
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize stream buffer
        self.buffer = StreamBuffer(
            max_size=sequence_length,
            feature_extractor=RealFeatureExtractor()
        )
        
        # Performance tracking
        self.latencies = deque(maxlen=1000)
        self.prediction_count = 0
        self.threat_count = 0
    
    def process_event(
        self, 
        event: Dict,
        return_components: bool = False
    ) -> Optional[ThreatPrediction]:
        """
        Process single event in real-time.
        
        Args:
            event: File system event dictionary
            return_components: Whether to return component scores
            
        Returns:
            ThreatPrediction if buffer is ready, None otherwise
        """
        start_time = time.perf_counter()
        
        # Add event to buffer
        buffer_ready = self.buffer.add_event(event)
        
        if not buffer_ready:
            return None
        
        # Get sequence from buffer
        sequence = self.buffer.get_sequence(self.sequence_length)
        if sequence is None:
            return None
        
        # Run inference
        with torch.no_grad():
            # Prepare input tensor (add batch dimension)
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Get prediction
            threat_score, components = self.model(x, return_components=return_components)
            threat_score = threat_score.cpu().item()
            
            is_threat = threat_score >= self.threshold
            confidence = abs(threat_score - 0.5) * 2  # Convert to 0-1 confidence
        
        # Calculate latency
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        self.latencies.append(latency_ms)
        
        # Update counters
        self.prediction_count += 1
        if is_threat:
            self.threat_count += 1
        
        # Build result
        result = ThreatPrediction(
            event_id=event.get('path', 'unknown'),
            timestamp=event.get('timestamp', time.time()),
            threat_score=threat_score,
            is_threat=is_threat,
            confidence=confidence,
            latency_ms=latency_ms
        )
        
        # Add component scores if requested
        if return_components and components:
            result.components = {
                k: v.cpu().item() if isinstance(v, torch.Tensor) else v
                for k, v in components.items()
                if k in ['dl_score', 'if_score', 'heuristic_score']
            }
        
        return result
    
    def process_batch(
        self, 
        events: List[Dict],
        return_components: bool = False
    ) -> List[ThreatPrediction]:
        """
        Process batch of events.
        
        Args:
            events: List of file system events
            return_components: Whether to return component scores
            
        Returns:
            List of ThreatPrediction results
        """
        results = []
        
        for event in events:
            prediction = self.process_event(event, return_components)
            if prediction is not None:
                results.append(prediction)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get current performance statistics.
        
        Returns:
            Dictionary with latency and throughput metrics
        """
        if not self.latencies:
            return {
                'avg_latency_ms': 0.0,
                'p50_latency_ms': 0.0,
                'p95_latency_ms': 0.0,
                'p99_latency_ms': 0.0,
                'max_latency_ms': 0.0,
                'total_predictions': 0,
                'threat_rate': 0.0
            }
        
        latencies_arr = np.array(self.latencies)
        
        return {
            'avg_latency_ms': float(np.mean(latencies_arr)),
            'p50_latency_ms': float(np.percentile(latencies_arr, 50)),
            'p95_latency_ms': float(np.percentile(latencies_arr, 95)),
            'p99_latency_ms': float(np.percentile(latencies_arr, 99)),
            'max_latency_ms': float(np.max(latencies_arr)),
            'total_predictions': self.prediction_count,
            'threat_rate': self.threat_count / max(1, self.prediction_count)
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.latencies.clear()
        self.prediction_count = 0
        self.threat_count = 0
    
    def clear_buffer(self):
        """Clear event buffer."""
        self.buffer.clear()
    
    @property
    def buffer_size(self) -> int:
        """Current buffer size."""
        return self.buffer.size
    
    @property
    def is_ready(self) -> bool:
        """Check if engine is ready for inference."""
        return self.buffer.is_ready(self.sequence_length)
