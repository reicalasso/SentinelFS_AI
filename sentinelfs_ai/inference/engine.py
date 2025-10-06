"""
Advanced production inference engine with explainability.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

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
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        feature_extractor: FeatureExtractor,
        threshold: float = 0.5, 
        enable_explainability: bool = True
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        self.enable_explainability = enable_explainability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Feature names for explanation
        self.feature_names = [
            'file_size_mb', 'access_hour', 'access_type', 
            'day_of_week', 'access_frequency', 'file_category', 
            'access_velocity'
        ]
    
    def _explain_features(self, x: torch.Tensor, normalized_x: np.ndarray) -> Dict[str, Any]:
        """
        Generate feature-level explanations.
        
        Args:
            x: Normalized input tensor
            normalized_x: Original normalized numpy array
            
        Returns:
            Dictionary with feature explanations
        """
        # Calculate feature statistics
        mean_features = normalized_x[0].mean(axis=0)
        max_features = normalized_x[0].max(axis=0)
        min_features = normalized_x[0].min(axis=0)
        
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
            'summary': []
        }
        
        # Get top 3 most important features
        top_indices = np.argsort(feature_importance)[-3:][::-1]
        
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            importance = float(feature_importance[idx])
            mean_val = float(mean_features[idx])
            
            explanation['important_features'][feature_name] = {
                'importance': importance,
                'mean_value': mean_val,
                'max_value': float(max_features[idx]),
                'min_value': float(min_features[idx])
            }
            
            # Generate human-readable summary
            if feature_name == 'access_hour':
                if mean_val < -0.5:  # Normalized value indicating off-hours
                    explanation['summary'].append("Off-hours access detected")
            elif feature_name == 'file_size_mb':
                if mean_val > 1.0:  # Large files
                    explanation['summary'].append("Unusually large file sizes")
            elif feature_name == 'access_velocity':
                if mean_val > 1.5:  # High velocity
                    explanation['summary'].append("Rapid access pattern detected")
            elif feature_name == 'access_frequency':
                if mean_val > 1.5:  # High frequency
                    explanation['summary'].append("High access frequency")
        
        return explanation
    
    def analyze(
        self, 
        access_sequence: np.ndarray, 
        return_attention: bool = None
    ) -> AnalysisResult:
        """
        Analyze a sequence of file access events with advanced features.
        
        Args:
            access_sequence: Numpy array of shape (seq_len, num_features)
            return_attention: Whether to return attention weights (None = auto-detect)
            
        Returns:
            AnalysisResult with comprehensive analysis
        """
        if return_attention is None:
            return_attention = self.enable_explainability
        
        with torch.no_grad():
            # Normalize features
            if len(access_sequence.shape) == 2:
                access_sequence = access_sequence.reshape(1, *access_sequence.shape)
            
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
            if self.enable_explainability and anomaly_detected:
                explanation = self._explain_features(x, normalized)
            
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
        parallel: bool = True
    ) -> List[AnalysisResult]:
        """
        Analyze multiple sequences in batch with optional parallelization.
        
        Args:
            sequences: List of access sequences
            parallel: Whether to process in parallel (faster for large batches)
            
        Returns:
            List of AnalysisResult objects
        """
        if not parallel or len(sequences) < 4:
            # Sequential processing for small batches
            results = []
            with torch.no_grad():
                for seq in sequences:
                    result = self.analyze(seq)
                    results.append(result)
            return results
        
        # Parallel batch processing
        results = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                
                # Ensure all sequences have same shape
                max_len = max(seq.shape[0] for seq in batch)
                padded_batch = []
                
                for seq in batch:
                    if len(seq.shape) == 2:
                        seq = seq.reshape(1, *seq.shape)
                    normalized = self.feature_extractor.transform(seq)
                    padded_batch.append(normalized[0])
                
                # Stack into batch tensor
                batch_tensor = torch.FloatTensor(np.array(padded_batch)).to(self.device)
                
                # Process batch
                outputs = self.model(batch_tensor)
                
                # Create results
                for j, output in enumerate(outputs):
                    score = output.item()
                    confidence = abs(score - self.threshold) / 0.5
                    confidence = min(confidence, 1.0)
                    anomaly_detected = score >= self.threshold
                    
                    results.append(AnalysisResult(
                        access_pattern_score=score,
                        behavior_normal=not anomaly_detected,
                        anomaly_detected=anomaly_detected,
                        confidence=confidence,
                        last_updated=datetime.now().isoformat(),
                        threat_score=score * 100
                    ))
        
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
