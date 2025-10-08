"""
Real-time production inference engine for SentinelFS integration.
Optimized for low latency (<25ms) threat detection.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import deque
import time
from threading import Lock

from ..models.hybrid_detector import HybridThreatDetector, LightweightThreatDetector
from ..data.real_feature_extractor import RealFeatureExtractor
from ..data_types import AnalysisResult, AnomalyType
from ..utils.logger import get_logger
from ..utils.device import get_device
from ..security import SecurityEngine

logger = get_logger(__name__)


class RealTimeInferenceEngine:
    """
    Production inference engine for real-time threat detection.
    
    Features:
    - Sub-25ms latency for threat detection
    - Stateful sequence management per user
    - Batch processing for efficiency
    - Model warming and optimization
    - Thread-safe operations
    - Performance monitoring
    - Automatic model updates
    """
    
    def __init__(
        self,
        model: HybridThreatDetector,
        feature_extractor: RealFeatureExtractor,
        sequence_length: int = 50,
        threat_threshold: float = 0.5,
        enable_caching: bool = True,
        enable_batching: bool = True,
        batch_timeout_ms: float = 5.0,
        max_batch_size: int = 32,
        enable_security_engine: bool = True
    ):
        """
        Initialize real-time inference engine.
        
        Args:
            model: Trained hybrid detector
            feature_extractor: Feature extraction system
            sequence_length: Length of sequences for analysis
            threat_threshold: Threshold for threat detection
            enable_caching: Enable result caching
            enable_batching: Enable request batching
            batch_timeout_ms: Max wait time for batching
            max_batch_size: Maximum batch size
            enable_security_engine: Enable advanced security engine
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.sequence_length = sequence_length
        self.threat_threshold = threat_threshold
        
        # Device setup
        self.device = get_device()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Optimize model for inference
        self._optimize_model()
        
        # Security Engine Integration (Phase 2.1)
        self.enable_security_engine = enable_security_engine
        self.security_engine = SecurityEngine() if enable_security_engine else None
        
        # User sequence buffers (thread-safe)
        self.user_sequences: Dict[str, deque] = {}
        self.sequence_lock = Lock()
        
        # Caching
        self.enable_caching = enable_caching
        self.cache: Dict[str, Tuple[AnalysisResult, float]] = {}
        self.cache_ttl = 60.0  # seconds
        
        # Batching
        self.enable_batching = enable_batching
        self.batch_timeout_ms = batch_timeout_ms
        self.max_batch_size = max_batch_size
        self.pending_requests = []
        self.batch_lock = Lock()
        
        # Performance monitoring
        self.performance_stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'avg_latency_ms': 0.0,
            'p95_latency_ms': 0.0,
            'p99_latency_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'threats_detected': 0,
            'false_positives': 0  # Requires feedback
        }
        self.latency_history = deque(maxlen=1000)
        
        logger.info("Real-time inference engine initialized")
        logger.info(f"Device: {self.device}, Sequence length: {sequence_length}")
    
    def analyze_event(
        self, 
        event: Dict[str, Any],
        return_explanation: bool = False
    ) -> AnalysisResult:
        """
        Analyze a single file system event for threats.
        
        Args:
            event: Event dictionary with file system operation details
            return_explanation: Include detailed explanation
            
        Returns:
            AnalysisResult with threat assessment
        """
        start_time = time.perf_counter()
        
        user_id = event.get('user_id', 'unknown')
        
        # Check cache
        if self.enable_caching:
            cache_key = self._generate_cache_key(event)
            cached_result = self._check_cache(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                return cached_result
            self.performance_stats['cache_misses'] += 1
        
        # Update user sequence
        with self.sequence_lock:
            if user_id not in self.user_sequences:
                self.user_sequences[user_id] = deque(maxlen=self.sequence_length)
            
            self.user_sequences[user_id].append(event)
            
            # Need at least sequence_length events for analysis
            if len(self.user_sequences[user_id]) < self.sequence_length:
                # Not enough data yet - return safe result
                return self._create_safe_result()
            
            # Get current sequence
            sequence = list(self.user_sequences[user_id])
        
        # Extract features
        features = self.feature_extractor.extract_from_sequence(sequence)
        
        # Perform inference
        result = self._perform_inference(
            features, 
            event, 
            return_explanation=return_explanation
        )
        
        # Update performance stats
        latency = (time.perf_counter() - start_time) * 1000  # ms
        self._update_performance_stats(latency, result.anomaly_detected)
        
        # Cache result
        if self.enable_caching:
            self._cache_result(cache_key, result)
        
        return result
    
    def analyze_batch(
        self,
        events: List[Dict[str, Any]]
    ) -> List[AnalysisResult]:
        """
        Analyze multiple events in batch for efficiency.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            List of AnalysisResults
        """
        start_time = time.perf_counter()
        
        results = []
        batch_features = []
        valid_indices = []
        
        # Prepare batch
        for idx, event in enumerate(events):
            user_id = event.get('user_id', 'unknown')
            
            # Update sequences
            with self.sequence_lock:
                if user_id not in self.user_sequences:
                    self.user_sequences[user_id] = deque(maxlen=self.sequence_length)
                
                self.user_sequences[user_id].append(event)
                
                if len(self.user_sequences[user_id]) >= self.sequence_length:
                    sequence = list(self.user_sequences[user_id])
                    features = self.feature_extractor.extract_from_sequence(sequence)
                    batch_features.append(features)
                    valid_indices.append(idx)
                else:
                    # Not enough data
                    results.append(self._create_safe_result())
        
        # Batch inference
        if batch_features:
            batch_tensor = torch.FloatTensor(np.array(batch_features)).to(self.device)
            
            with torch.no_grad():
                scores, components = self.model(batch_tensor, return_components=True)
            
            # Create results for valid indices
            for i, idx in enumerate(valid_indices):
                score = scores[i].item()
                is_threat = score > self.threat_threshold
                
                result = AnalysisResult(
                    access_pattern_score=score,
                    behavior_normal=not is_threat,
                    anomaly_detected=is_threat,
                    confidence=abs(score - 0.5) * 2,
                    last_updated=datetime.now().isoformat(),
                    threat_score=score,
                    anomaly_type=self._determine_anomaly_type(
                        components, i
                    ) if is_threat else None
                )
                
                # Insert at correct position
                results.insert(idx, result)
        
        # Update performance
        latency = (time.perf_counter() - start_time) * 1000
        avg_latency = latency / len(events) if events else 0
        self._update_performance_stats(avg_latency, any(r.anomaly_detected for r in results))
        
        return results
    
    def _perform_inference(
        self,
        features: np.ndarray,
        event: Dict[str, Any],
        return_explanation: bool = False
    ) -> AnalysisResult:
        """Perform actual model inference with security engine integration."""
        # Prepare input
        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # AI Model Inference
        with torch.no_grad():
            score, components = self.model(input_tensor, return_components=True)

        score_value = score.item()
        is_threat = score_value > self.threat_threshold
        confidence = abs(score_value - 0.5) * 2  # Distance from decision boundary

        # Security Engine Analysis (Phase 2.1)
        security_result = None
        if self.enable_security_engine and self.security_engine:
            try:
                # Analyze file if path is available
                file_path = event.get('file_path')
                if file_path:
                    security_result = self.security_engine.analyze_file(
                        file_path=file_path,
                        ai_score=score_value
                    )
                    # Update threat assessment with security engine
                    if security_result.final_decision:
                        is_threat = True
                        # Adjust confidence based on security engine
                        confidence = max(confidence, security_result.combined_score)
                        score_value = max(score_value, security_result.combined_score)
            except Exception as e:
                logger.warning(f"Security engine analysis failed: {e}")

        # Determine anomaly type if threat detected
        anomaly_type = None
        anomaly_type_confidence = None
        if is_threat:
            anomaly_type, anomaly_type_confidence = self._determine_anomaly_type(
                components, 0
            )
        
        # Generate explanation if requested
        explanation = None
        if return_explanation and components is not None:
            explanation = self._generate_explanation(
                features, components, event, score_value
            )
        
        # Create result
        result = AnalysisResult(
            access_pattern_score=score_value,
            behavior_normal=not is_threat,
            anomaly_detected=is_threat,
            confidence=confidence,
            last_updated=datetime.now().isoformat(),
            threat_score=score_value,
            anomaly_type=anomaly_type,
            anomaly_type_confidence=anomaly_type_confidence,
            attention_weights=components['attention_weights'][0].squeeze().cpu().numpy().tolist() if components else None,
            explanation=explanation,
            # Phase 2.1: Security Engine Integration
            security_engine_enabled=self.enable_security_engine,
            security_score=security_result.combined_score if security_result else None,
            security_threat_level=security_result.threat_level.value if security_result else None,
            security_details={
                'ai_score': security_result.ai_score if security_result else None,
                'security_score': security_result.security_score if security_result else None,
                'correlation_factors': security_result.correlation_factors if security_result else None,
                'results': [r.__dict__ for r in security_result.results] if security_result else None
            } if security_result else None,
            detection_methods=[m.value for m in security_result.detection_methods] if security_result else None
        )
        
        return result
    
    def _determine_anomaly_type(
        self, 
        components: Dict[str, Any],
        batch_idx: int
    ) -> Tuple[Optional[str], Optional[float]]:
        """Determine the type of anomaly from component scores."""
        # Analyze which component contributed most
        heuristic_score = components['heuristic_score'][batch_idx].item()
        dl_score = components['dl_score'][batch_idx].item()
        if_score = components['if_score'][batch_idx].item()
        
        # Determine primary indicator
        scores = {
            'heuristic': heuristic_score,
            'deep_learning': dl_score,
            'anomaly_detection': if_score
        }
        primary_component = max(scores, key=scores.get)
        max_score = scores[primary_component]
        
        # Map to anomaly types
        if primary_component == 'heuristic' and heuristic_score > 0.7:
            return 'RANSOMWARE', heuristic_score
        elif if_score > 0.7:
            return 'ANOMALOUS_PATTERN', if_score
        elif dl_score > 0.7:
            return 'SUSPICIOUS_BEHAVIOR', dl_score
        else:
            return 'POTENTIAL_THREAT', max_score
    
    def _generate_explanation(
        self,
        features: np.ndarray,
        components: Dict[str, Any],
        event: Dict[str, Any],
        score: float
    ) -> Dict[str, Any]:
        """Generate human-readable explanation."""
        # Get feature names and values
        feature_names = self.feature_extractor.get_feature_names()
        feature_values = features[-1]  # Last timestep
        
        # Find top contributing features
        feature_importance = np.abs(feature_values - 0.5)
        top_indices = np.argsort(feature_importance)[-5:][::-1]
        
        top_features = []
        for idx in top_indices:
            if idx < len(feature_names):
                top_features.append({
                    'feature': feature_names[idx],
                    'value': float(feature_values[idx]),
                    'importance': float(feature_importance[idx])
                })
        
        # Component contributions
        component_scores = {
            'heuristic_rules': float(components['heuristic_score'][0].item()),
            'deep_learning': float(components['dl_score'][0].item()),
            'anomaly_detection': float(components['if_score'][0].item())
        }
        
        # Generate reason
        reasons = []
        if component_scores['heuristic_rules'] > 0.6:
            reasons.append("Known attack pattern detected by heuristic rules")
        if component_scores['anomaly_detection'] > 0.6:
            reasons.append("Statistical anomaly in behavior pattern")
        if component_scores['deep_learning'] > 0.6:
            reasons.append("Suspicious temporal access pattern")
        
        return {
            'score': score,
            'component_scores': component_scores,
            'top_features': top_features,
            'primary_reasons': reasons,
            'file_path': event.get('file_path', 'unknown'),
            'operation': event.get('operation', 'unknown')
        }
    
    def _optimize_model(self):
        """Optimize model for inference."""
        # Set to eval mode
        self.model.eval()
        
        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Warm up model
        logger.info("Warming up model...")
        dummy_input = torch.randn(
            1, self.sequence_length, 
            self.feature_extractor.get_num_features(),
            device=self.device
        )
        
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        logger.info("Model optimization complete")
    
    def _create_safe_result(self) -> AnalysisResult:
        """Create a safe default result when insufficient data."""
        return AnalysisResult(
            access_pattern_score=0.0,
            behavior_normal=True,
            anomaly_detected=False,
            confidence=0.0,
            last_updated=datetime.now().isoformat(),
            threat_score=0.0,
            anomaly_type=None
        )
    
    def _generate_cache_key(self, event: Dict[str, Any]) -> str:
        """Generate cache key from event."""
        user_id = event.get('user_id', 'unknown')
        file_path = event.get('file_path', '')
        operation = event.get('operation', '')
        timestamp = event.get('timestamp', '')
        
        return f"{user_id}:{file_path}:{operation}:{timestamp}"
    
    def _check_cache(self, key: str) -> Optional[AnalysisResult]:
        """Check cache for existing result."""
        if key in self.cache:
            result, cached_time = self.cache[key]
            if time.time() - cached_time < self.cache_ttl:
                return result
            else:
                # Expired
                del self.cache[key]
        return None
    
    def _cache_result(self, key: str, result: AnalysisResult):
        """Cache analysis result."""
        self.cache[key] = (result, time.time())
        
        # Limit cache size
        if len(self.cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k][1]
            )[:100]
            for k in oldest_keys:
                del self.cache[k]
    
    def _update_performance_stats(self, latency_ms: float, is_threat: bool):
        """Update performance statistics."""
        self.performance_stats['total_inferences'] += 1
        self.performance_stats['total_time'] += latency_ms
        self.latency_history.append(latency_ms)
        
        if is_threat:
            self.performance_stats['threats_detected'] += 1
        
        # Update averages
        self.performance_stats['avg_latency_ms'] = (
            self.performance_stats['total_time'] / 
            self.performance_stats['total_inferences']
        )
        
        # Update percentiles
        if len(self.latency_history) > 20:
            sorted_latencies = sorted(self.latency_history)
            self.performance_stats['p95_latency_ms'] = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            self.performance_stats['p99_latency_ms'] = sorted_latencies[int(len(sorted_latencies) * 0.99)]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add cache statistics
        if self.enable_caching:
            total_cache_requests = stats['cache_hits'] + stats['cache_misses']
            stats['cache_hit_rate'] = (
                stats['cache_hits'] / total_cache_requests 
                if total_cache_requests > 0 else 0.0
            )
        
        # Add detection rate
        stats['threat_detection_rate'] = (
            stats['threats_detected'] / stats['total_inferences']
            if stats['total_inferences'] > 0 else 0.0
        )
        
        return stats
    
    def reset_user_history(self, user_id: str):
        """Reset sequence history for a user."""
        with self.sequence_lock:
            if user_id in self.user_sequences:
                del self.user_sequences[user_id]
                logger.debug(f"Reset history for user {user_id}")
    
    def clear_cache(self):
        """Clear result cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def export_performance_metrics(self) -> Dict[str, Any]:
        """Export metrics in Prometheus format."""
        stats = self.get_performance_stats()
        
        return {
            'sentinel_ai_inferences_total': stats['total_inferences'],
            'sentinel_ai_threats_detected_total': stats['threats_detected'],
            'sentinel_ai_inference_latency_ms': {
                'avg': stats['avg_latency_ms'],
                'p95': stats['p95_latency_ms'],
                'p99': stats['p99_latency_ms']
            },
            'sentinel_ai_cache_hit_rate': stats.get('cache_hit_rate', 0.0),
            'sentinel_ai_threat_detection_rate': stats['threat_detection_rate']
        }
