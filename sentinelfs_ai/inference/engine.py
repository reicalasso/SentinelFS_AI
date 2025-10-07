"""
Advanced production inference engine with explainability.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import warnings

from ..data_types import AnalysisResult, AnomalyType
from ..data.feature_extractor import FeatureExtractor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """
    Advanced production inference engine with explainability.
    
    Features:
    - Anomaly detection with confidence scores
    - Anomaly type classification
    - Attention weight visualization
    - Feature importance analysis
    - Batch processing support
    - Performance monitoring
    - Real-time explainability
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        feature_extractor: FeatureExtractor,
        threshold: float = 0.5, 
        enable_explainability: bool = True,
        enable_performance_monitoring: bool = True,
        performance_window: int = 100
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        self.enable_explainability = enable_explainability
        self.enable_performance_monitoring = enable_performance_monitoring
        self.performance_window = performance_window
        
        # Performance monitoring
        self.inference_times = []
        self.anomaly_detection_rate = 0.0
        self.confidence_scores = []
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Feature names for explanation
        self.feature_names = [
            'file_size_mb', 'access_hour', 'access_type', 
            'day_of_week', 'access_frequency', 'file_category', 
            'access_velocity'
        ]
    
    def _explain_features(self, x: torch.Tensor, normalized_x: np.ndarray, raw_x: np.ndarray) -> Dict[str, Any]:
        """
        Generate feature-level explanations with more detailed analysis.
        
        Args:
            x: Normalized input tensor
            normalized_x: Normalized numpy array
            raw_x: Original (unnormalized) numpy array
            
        Returns:
            Dictionary with feature explanations
        """
        # Calculate feature statistics
        mean_features = normalized_x[0].mean(axis=0)
        max_features = normalized_x[0].max(axis=0)
        min_features = normalized_x[0].min(axis=0)
        std_features = normalized_x[0].std(axis=0)
        
        # Calculate raw feature statistics for human-readable explanations
        raw_mean_features = raw_x[0].mean(axis=0) if raw_x is not None else mean_features
        raw_max_features = raw_x[0].max(axis=0) if raw_x is not None else max_features
        
        # Get feature importance using gradient-based method
        was_training = self.model.training
        self.model.train()  # Enable gradients temporarily
        
        x_grad = x.clone().requires_grad_(True)
        output = self.model(x_grad)
        
        # Compute gradients
        if output.requires_grad:
            output.backward()
            feature_importance = x_grad.grad.abs().mean(dim=1).squeeze().cpu().numpy()
        else:
            # Fallback: use variance as importance measure
            feature_importance = normalized_x[0].var(axis=0)
        
        # Restore model mode
        if not was_training:
            self.model.eval()
        
        # Create explanation
        explanation = {
            'important_features': {},
            'summary': [],
            'anomaly_pattern': 'normal',
            'feature_contributions': {},
            'risk_factors': []
        }
        
        # Get all features' importance and analyze them
        for idx, feature_name in enumerate(self.feature_names):
            importance = float(feature_importance[idx])
            mean_val = float(mean_features[idx])
            raw_mean_val = float(raw_mean_features[idx]) if raw_x is not None else mean_val
            std_val = float(std_features[idx])
            
            explanation['feature_contributions'][feature_name] = float(importance)
            
            # Add more specific explanations based on feature values
            if feature_name == 'access_hour':
                # Check for off-hours access (assuming normalized values around 0)
                if raw_mean_val < 8 or raw_mean_val > 18:  # Non-business hours
                    explanation['risk_factors'].append(f"Access during off-hours (avg hour: {raw_mean_val:.1f})")
                    explanation['summary'].append(f"Off-hours access detected (hour: {raw_mean_val:.1f})")
                elif raw_mean_val > 22 or raw_mean_val < 4:  # Very late night/early morning
                    explanation['risk_factors'].append(f"Very late/early access (hour: {raw_mean_val:.1f})")
                    explanation['summary'].append(f"Very unusual timing (hour: {raw_mean_val:.1f})")
            elif feature_name == 'file_size_mb':
                if raw_mean_val > 100:  # Large files
                    explanation['risk_factors'].append(f"Large file access (avg: {raw_mean_val:.1f}MB)")
                    explanation['summary'].append(f"Unusually large file sizes ({raw_mean_val:.1f}MB)")
            elif feature_name == 'access_velocity':
                if raw_mean_val > 10:  # High velocity
                    explanation['risk_factors'].append(f"High access velocity ({raw_mean_val:.1f}/min)")
                    explanation['summary'].append(f"Rapid access pattern ({raw_mean_val:.1f}/min)")
            elif feature_name == 'access_frequency':
                if raw_mean_val > 50:  # High frequency
                    explanation['risk_factors'].append(f"High access frequency ({raw_mean_val:.1f}/hr)")
                    explanation['summary'].append(f"High access frequency ({raw_mean_val:.1f}/hr)")
        
        # Determine anomaly pattern type
        risk_score = len(explanation['risk_factors'])
        if risk_score >= 3:
            explanation['anomaly_pattern'] = 'high_risk'
        elif risk_score == 2:
            explanation['anomaly_pattern'] = 'moderate_risk'
        elif risk_score == 1:
            explanation['anomaly_pattern'] = 'low_risk'
        else:
            explanation['anomaly_pattern'] = 'normal'
        
        # Get top 3 most important features
        top_indices = np.argsort(feature_importance)[-3:][::-1]
        
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            importance = float(feature_importance[idx])
            mean_val = float(mean_features[idx])
            
            explanation['important_features'][feature_name] = {
                'importance': importance,
                'mean_value': mean_val,
                'raw_mean_value': float(raw_mean_features[idx]),
                'max_value': float(max_features[idx]),
                'min_value': float(min_features[idx]),
                'std_value': float(std_features[idx])
            }
        
        return explanation
    
    def analyze(
        self, 
        access_sequence: np.ndarray, 
        return_attention: bool = None,
        include_raw_features: bool = True
    ) -> AnalysisResult:
        """
        Analyze a sequence of file access events with advanced features.
        
        Args:
            access_sequence: Numpy array of shape (seq_len, num_features)
            return_attention: Whether to return attention weights (None = auto-detect)
            include_raw_features: Whether to include raw features in explanations
            
        Returns:
            AnalysisResult with comprehensive analysis
        """
        import time
        
        if return_attention is None:
            return_attention = self.enable_explainability
        
        start_time = time.time()
        
        with torch.no_grad():
            # Normalize features
            if len(access_sequence.shape) == 2:
                access_sequence = access_sequence.reshape(1, *access_sequence.shape)
            
            # Store raw features for explanations
            raw_features = access_sequence.copy() if include_raw_features else None
            
            normalized = self.feature_extractor.transform(access_sequence)
            x = torch.FloatTensor(normalized).to(self.device)
            
            # Get prediction with attention
            if return_attention and hasattr(self.model, 'use_attention') and self.model.use_attention:
                output, attention_weights = self.model(x, return_attention=True)
                attention_weights = attention_weights.cpu().numpy().tolist()[0]
            else:
                output = self.model(x)
                attention_weights = None
            
            score = output.item()
            
            # Get anomaly type if model supports it
            anomaly_type_id = None
            anomaly_type_conf = None
            if hasattr(self.model, 'predict_anomaly_type'):
                type_probs = self.model.predict_anomaly_type(x)
                anomaly_type_id = type_probs.argmax(dim=1).item()
                anomaly_type_conf = type_probs.max(dim=1).values.item()
            
            # Calculate confidence (distance from threshold)
            confidence = abs(score - self.threshold) / 0.5
            confidence = min(confidence, 1.0)
            
            anomaly_detected = score >= self.threshold
            
            # Generate explanation if enabled and anomaly detected
            explanation = None
            if self.enable_explainability:
                explanation = self._explain_features(x, normalized, raw_features)
            
            # Performance monitoring
            if self.enable_performance_monitoring:
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                if len(self.inference_times) > self.performance_window:
                    self.inference_times.pop(0)
                
                self.confidence_scores.append(score)
                if len(self.confidence_scores) > self.performance_window:
                    self.confidence_scores.pop(0)
                
                # Update anomaly detection rate
                recent_predictions = self.confidence_scores[-self.performance_window:]
                if recent_predictions:
                    self.anomaly_detection_rate = np.mean([1 if pred >= self.threshold else 0 for pred in recent_predictions])
            
            return AnalysisResult(
                access_pattern_score=score,
                behavior_normal=not anomaly_detected,
                anomaly_detected=anomaly_detected,
                confidence=confidence,
                last_updated=datetime.now().isoformat(),
                threat_score=score * 100,  # Convert to 0-100 scale
                anomaly_type=AnomalyType.get_name(anomaly_type_id) if anomaly_type_id is not None else None,
                anomaly_type_confidence=anomaly_type_conf,
                attention_weights=attention_weights,
                explanation=explanation
            )
    
    def batch_analyze(
        self, 
        sequences: List[np.ndarray], 
        parallel: bool = True,
        batch_size: int = 32
    ) -> List[AnalysisResult]:
        """
        Analyze multiple sequences in batch with optional parallelization.
        
        Args:
            sequences: List of access sequences
            parallel: Whether to process in parallel (faster for large batches)
            batch_size: Size of processing batches
            
        Returns:
            List of AnalysisResult objects
        """
        if not parallel or len(sequences) < 4:
            # Sequential processing for small batches
            results = []
            for seq in sequences:
                result = self.analyze(seq)
                results.append(result)
            return results
        
        # Batch processing
        results = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            
            # Process batch
            batch_results = []
            for seq in batch:
                result = self.analyze(seq)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def get_attention_heatmap(self, access_sequence: np.ndarray) -> Optional[np.ndarray]:
        """
        Get attention heatmap for visualization.
        
        Args:
            access_sequence: Access sequence to analyze
            
        Returns:
            Attention weights array or None if not available
        """
        result = self.analyze(access_sequence, return_attention=True)
        if result.attention_weights:
            return np.array(result.attention_weights)
        return None
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the inference engine.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {
                'avg_inference_time_ms': 0.0,
                'min_inference_time_ms': 0.0,
                'max_inference_time_ms': 0.0,
                'std_inference_time_ms': 0.0,
                'anomaly_detection_rate': 0.0,
                'samples_processed': 0
            }
        
        inference_times_ms = [t * 1000 for t in self.inference_times]
        return {
            'avg_inference_time_ms': float(np.mean(inference_times_ms)),
            'min_inference_time_ms': float(np.min(inference_times_ms)),
            'max_inference_time_ms': float(np.max(inference_times_ms)),
            'std_inference_time_ms': float(np.std(inference_times_ms)),
            'anomaly_detection_rate': self.anomaly_detection_rate,
            'samples_processed': len(self.confidence_scores)
        }
    
    def reset_performance_monitoring(self):
        """Reset all performance monitoring statistics."""
        self.inference_times = []
        self.anomaly_detection_rate = 0.0
        self.confidence_scores = []
    
    def explain_prediction(
        self,
        access_sequence: np.ndarray,
        method: str = 'gradient'  # 'gradient', 'integrated', 'attention'
    ) -> Dict[str, Any]:
        """
        Generate a detailed explanation for a prediction using various methods.
        
        Args:
            access_sequence: Access sequence to explain
            method: Explanation method to use
            
        Returns:
            Dictionary with detailed explanation
        """
        if method == 'gradient':
            return self._explain_features(
                torch.FloatTensor(self.feature_extractor.transform(access_sequence.reshape(1, *access_sequence.shape))).to(self.device),
                self.feature_extractor.transform(access_sequence.reshape(1, *access_sequence.shape)),
                access_sequence
            )
        elif method == 'attention' and hasattr(self.model, 'use_attention') and self.model.use_attention:
            # Get attention-based explanation
            with torch.no_grad():
                x = torch.FloatTensor(self.feature_extractor.transform(access_sequence.reshape(1, *access_sequence.shape))).to(self.device)
                _, attention_weights = self.model(x, return_attention=True)
                
                # Analyze which time steps were most attended to
                attention_weights_np = attention_weights.cpu().numpy()
                top_attention_idx = np.argmax(attention_weights_np)
                top_attention_value = attention_weights_np[top_attention_idx]
                
                return {
                    'method': 'attention',
                    'most_important_timestep': int(top_attention_idx),
                    'attention_weight': float(top_attention_value),
                    'attention_distribution': attention_weights_np.tolist(),
                    'explanation': f"Model focused most on timestep {top_attention_idx} (weight: {top_attention_value:.3f})"
                }
        else:
            # Fallback to gradient method
            return self._explain_features(
                torch.FloatTensor(self.feature_extractor.transform(access_sequence.reshape(1, *access_sequence.shape))).to(self.device),
                self.feature_extractor.transform(access_sequence.reshape(1, *access_sequence.shape)),
                access_sequence
            )
    
    def predict_anomaly_type_detailed(self, access_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Get detailed anomaly type prediction with confidence breakdown.
        
        Args:
            access_sequence: Access sequence to analyze
            
        Returns:
            Dictionary with detailed anomaly type predictions
        """
        with torch.no_grad():
            x = torch.FloatTensor(self.feature_extractor.transform(access_sequence.reshape(1, *access_sequence.shape))).to(self.device)
            
            if hasattr(self.model, 'predict_anomaly_type'):
                type_probs = self.model.predict_anomaly_type(x)
                type_probs_np = type_probs.cpu().numpy()[0]
                
                # Get all anomaly type probabilities
                anomaly_type_probs = {
                    AnomalyType.get_name(i): float(prob) 
                    for i, prob in enumerate(type_probs_np)
                }
                
                predicted_type_id = int(np.argmax(type_probs_np))
                predicted_type_name = AnomalyType.get_name(predicted_type_id)
                confidence = float(np.max(type_probs_np))
                
                return {
                    'predicted_type': predicted_type_name,
                    'confidence': confidence,
                    'all_probabilities': anomaly_type_probs,
                    'type_id': predicted_type_id
                }
            else:
                return {
                    'predicted_type': None,
                    'confidence': 0.0,
                    'all_probabilities': {},
                    'type_id': None
                }
