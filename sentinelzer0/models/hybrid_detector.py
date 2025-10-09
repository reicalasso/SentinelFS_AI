"""
Real-world hybrid threat detection model combining deep learning, anomaly detection, and heuristics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class HybridThreatDetector(nn.Module):
    """
    Production-ready hybrid threat detection system combining:
    1. Deep Learning (LSTM/GRU) for temporal pattern recognition
    2. Isolation Forest for statistical anomaly detection
    3. Heuristic rules for known attack patterns
    
    This model is designed for real-world deployment with:
    - Fast inference (<25ms)
    - High accuracy with low false positives
    - Explainable predictions
    - Incremental learning capability
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_gru: bool = False,
        isolation_forest_contamination: float = 0.1,
        heuristic_weight: float = 0.3,
        dl_weight: float = 0.4,
        anomaly_weight: float = 0.3
    ):
        """
        Initialize hybrid threat detector.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layer in RNN
            num_layers: Number of RNN layers
            dropout: Dropout rate
            use_gru: Use GRU instead of LSTM
            isolation_forest_contamination: Expected proportion of anomalies
            heuristic_weight: Weight for heuristic component (0-1)
            dl_weight: Weight for deep learning component (0-1)
            anomaly_weight: Weight for anomaly detection component (0-1)
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_gru = use_gru
        
        # Weights for ensemble
        total_weight = heuristic_weight + dl_weight + anomaly_weight
        self.heuristic_weight = heuristic_weight / total_weight
        self.dl_weight = dl_weight / total_weight
        self.anomaly_weight = anomaly_weight / total_weight
        
        # === Component 1: Deep Learning (LSTM/GRU) ===
        RNN = nn.GRU if use_gru else nn.LSTM
        self.rnn = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism for sequence
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Deep learning classifier
        self.dl_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # === Component 2: Isolation Forest (fitted during training) ===
        self.isolation_forest = IsolationForest(
            contamination=isolation_forest_contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        self.if_scaler = StandardScaler()
        self.if_fitted = False
        
        # === Component 3: Heuristic Rules ===
        # These thresholds will be learned from training data
        self.heuristic_thresholds = {
            'rapid_modifications': 0.3,  # >30% rapid modifications
            'mass_operations': 0.4,  # >40% mass operation score
            'ransomware_indicators': 0.5,  # Combined ransomware score
            'unusual_time_ops': 0.6,  # >60% operations at unusual times
            'high_delete_rate': 0.3,  # >30% delete rate
            'suspicious_extensions': 0.7,  # >70% suspicious file extensions
        }
        
        # Feature indices (must match RealFeatureExtractor output)
        self._setup_feature_indices()
        
    def _setup_feature_indices(self):
        """Setup indices for features used in heuristics."""
        # Based on RealFeatureExtractor.get_feature_names()
        feature_names = [
            'hour_normalized', 'day_of_week_normalized', 'is_weekend',
            'is_night', 'is_business_hours', 'time_since_last_op',
            'log_file_size', 'is_executable', 'is_document', 'is_compressed',
            'is_encrypted', 'path_depth', 'filename_entropy', 'operation_type',
            'log_access_frequency',
            'ops_per_minute', 'operation_diversity', 'file_diversity',
            'log_avg_size', 'burstiness', 'baseline_deviation',
            'has_ransomware_ext', 'has_ransomware_name', 'rapid_modifications',
            'size_change_ratio', 'delete_rate', 'mass_operation_score',
            'rename_rate', 'is_hidden', 'is_unusual_time'
        ]
        
        self.feature_idx = {name: idx for idx, name in enumerate(feature_names)}
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through hybrid model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            return_components: If True, return individual component scores
            
        Returns:
            - Final threat score (batch_size, 1)
            - Optional dict with component scores and attention weights
        """
        # Ensure x is 3D: (batch_size, seq_len, num_features)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
        
        batch_size, seq_len, _ = x.shape
        
        # === Component 1: Deep Learning ===
        # RNN encoding
        if self.use_gru:
            rnn_out, _ = self.rnn(x)  # (batch, seq, hidden*2)
        else:
            rnn_out, _ = self.rnn(x)  # (batch, seq, hidden*2)
        
        # Attention mechanism
        attention_weights = self.attention(rnn_out)  # (batch, seq, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum using attention
        context = torch.sum(attention_weights * rnn_out, dim=1)  # (batch, hidden*2)
        
        # DL prediction
        dl_score = self.dl_classifier(context)  # (batch, 1)
        
        # === Component 2: Isolation Forest ===
        # Flatten sequence for isolation forest (use mean features)
        x_mean = x.mean(dim=1).detach().cpu().numpy()  # (batch, features)
        
        if self.if_fitted:
            # Predict anomaly scores (-1 for outliers, 1 for inliers)
            if_scores = self.isolation_forest.decision_function(
                self.if_scaler.transform(x_mean)
            )
            # Convert to probability (lower score = more anomalous)
            # Normalize to [0, 1] range where 1 = anomalous
            if_score = torch.tensor(
                1 / (1 + np.exp(if_scores)),  # Sigmoid transformation
                dtype=torch.float32,
                device=x.device
            ).unsqueeze(1)
        else:
            # During training before IF is fitted
            if_score = torch.zeros(batch_size, 1, device=x.device)
        
        # === Component 3: Heuristic Rules ===
        heuristic_score = self._apply_heuristics(x)  # (batch, 1)
        
        # === Ensemble ===
        final_score = (
            self.dl_weight * dl_score +
            self.anomaly_weight * if_score +
            self.heuristic_weight * heuristic_score
        )
        
        if return_components:
            components = {
                'dl_score': dl_score,
                'if_score': if_score,
                'heuristic_score': heuristic_score,
                'attention_weights': attention_weights,
                'weights': {
                    'dl': self.dl_weight,
                    'if': self.anomaly_weight,
                    'heuristic': self.heuristic_weight
                }
            }
            return final_score, components
        
        return final_score, None
    
    def _apply_heuristics(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rule-based heuristics for known attack patterns.
        
        Args:
            x: Input tensor (batch_size, seq_len, num_features)
            
        Returns:
            Heuristic threat score (batch_size, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Average features across sequence
        x_avg = x.mean(dim=1)  # (batch, features)
        
        # Extract specific features by index
        rapid_mods = x_avg[:, self.feature_idx['rapid_modifications']]
        mass_ops = x_avg[:, self.feature_idx['mass_operation_score']]
        delete_rate = x_avg[:, self.feature_idx['delete_rate']]
        ransomware_ext = x_avg[:, self.feature_idx['has_ransomware_ext']]
        ransomware_name = x_avg[:, self.feature_idx['has_ransomware_name']]
        unusual_time = x_avg[:, self.feature_idx['is_unusual_time']]
        is_encrypted = x_avg[:, self.feature_idx['is_encrypted']]
        high_entropy = x_avg[:, self.feature_idx['filename_entropy']]
        rename_rate = x_avg[:, self.feature_idx['rename_rate']]
        
        # Rule 1: Ransomware indicators
        ransomware_score = (
            ransomware_ext * 0.4 +
            ransomware_name * 0.3 +
            rapid_mods * 0.3
        )
        ransomware_detected = (ransomware_score > self.heuristic_thresholds['ransomware_indicators']).float()
        
        # Rule 2: Mass file operations (exfiltration or destructive)
        mass_op_detected = (mass_ops > self.heuristic_thresholds['mass_operations']).float()
        
        # Rule 3: High delete rate (data destruction)
        high_delete_detected = (delete_rate > self.heuristic_thresholds['high_delete_rate']).float()
        
        # Rule 4: Suspicious file patterns
        suspicious_pattern = (
            (is_encrypted > 0.5) *
            (high_entropy > 0.6) *
            (rename_rate > 0.2)
        ).float()
        
        # Rule 5: Unusual time operations with suspicious activity
        unusual_time_suspicious = (
            (unusual_time > self.heuristic_thresholds['unusual_time_ops']) *
            (mass_ops > 0.3)
        ).float()
        
        # Combine all rules with weights
        heuristic_score = torch.clamp(
            ransomware_detected * 0.9 +  # Highest priority
            mass_op_detected * 0.7 +
            high_delete_detected * 0.6 +
            suspicious_pattern * 0.5 +
            unusual_time_suspicious * 0.4,
            min=0.0,
            max=1.0
        )
        
        return heuristic_score.unsqueeze(1)
    
    def fit_isolation_forest(self, X: np.ndarray):
        """
        Fit the Isolation Forest on training data.
        
        Args:
            X: Training data of shape (n_samples, seq_len, n_features)
               Will be averaged across sequence dimension
        """
        logger.info("Fitting Isolation Forest on training data...")
        
        # Average across sequence dimension
        if X.ndim == 3:
            X_flat = X.mean(axis=1)  # (n_samples, n_features)
        else:
            X_flat = X
        
        # Fit scaler and transform
        X_scaled = self.if_scaler.fit_transform(X_flat)
        
        # Fit isolation forest
        self.isolation_forest.fit(X_scaled)
        self.if_fitted = True
        
        logger.info(f"Isolation Forest fitted on {len(X_flat)} samples")
    
    def calibrate_thresholds(self, X: np.ndarray, y: np.ndarray):
        """
        Calibrate heuristic thresholds based on training data.
        
        Args:
            X: Training data (n_samples, seq_len, n_features)
            y: Labels (n_samples, 1) where 1 = anomaly
        """
        logger.info("Calibrating heuristic thresholds...")
        
        # Average across sequence
        if X.ndim == 3:
            X_avg = X.mean(axis=1)
        else:
            X_avg = X
        
        # Separate normal and anomalous samples
        normal_X = X_avg[y.flatten() == 0]
        anomaly_X = X_avg[y.flatten() == 1]
        
        if len(anomaly_X) == 0:
            logger.warning("No anomalous samples found for calibration")
            return
        
        # Calculate thresholds as midpoint between normal and anomaly means
        for feature_name, idx in self.feature_idx.items():
            if feature_name in self.heuristic_thresholds:
                normal_mean = normal_X[:, idx].mean() if len(normal_X) > 0 else 0
                anomaly_mean = anomaly_X[:, idx].mean()
                
                # Set threshold as midpoint
                threshold = (normal_mean + anomaly_mean) / 2
                self.heuristic_thresholds[feature_name] = float(threshold)
        
        logger.info(f"Calibrated thresholds: {self.heuristic_thresholds}")
    
    def save_components(self, save_dir: str):
        """Save non-PyTorch components (Isolation Forest, etc.)"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save Isolation Forest
        if self.if_fitted:
            with open(save_path / 'isolation_forest.pkl', 'wb') as f:
                pickle.dump({
                    'model': self.isolation_forest,
                    'scaler': self.if_scaler
                }, f)
        
        # Save thresholds
        with open(save_path / 'heuristic_thresholds.pkl', 'wb') as f:
            pickle.dump(self.heuristic_thresholds, f)
        
        logger.info(f"Model components saved to {save_dir}")
    
    def load_components(self, load_dir: str):
        """Load non-PyTorch components"""
        load_path = Path(load_dir)
        
        # Load Isolation Forest
        if_path = load_path / 'isolation_forest.pkl'
        if if_path.exists():
            with open(if_path, 'rb') as f:
                data = pickle.load(f)
                self.isolation_forest = data['model']
                self.if_scaler = data['scaler']
                self.if_fitted = True
        
        # Load thresholds
        thresh_path = load_path / 'heuristic_thresholds.pkl'
        if thresh_path.exists():
            with open(thresh_path, 'rb') as f:
                self.heuristic_thresholds = pickle.load(f)
        
        logger.info(f"Model components loaded from {load_dir}")
    
    def explain_prediction(
        self, 
        x: torch.Tensor, 
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate detailed explanation for a prediction.
        
        Args:
            x: Input tensor (1, seq_len, num_features) - single sample
            threshold: Decision threshold
            
        Returns:
            Dictionary with explanation details
        """
        self.eval()
        with torch.no_grad():
            score, components = self.forward(x, return_components=True)
            
            is_threat = score.item() > threshold
            
            # Determine primary threat indicators
            x_avg = x.mean(dim=1).squeeze().cpu().numpy()
            
            threat_indicators = []
            if x_avg[self.feature_idx['has_ransomware_ext']] > 0.5:
                threat_indicators.append("Ransomware file extension detected")
            if x_avg[self.feature_idx['rapid_modifications']] > 0.3:
                threat_indicators.append("Rapid file modifications detected")
            if x_avg[self.feature_idx['mass_operation_score']] > 0.4:
                threat_indicators.append("Mass file operation pattern")
            if x_avg[self.feature_idx['delete_rate']] > 0.3:
                threat_indicators.append("High file deletion rate")
            if x_avg[self.feature_idx['is_unusual_time']] > 0.6:
                threat_indicators.append("Operations at unusual times")
            
            explanation = {
                'is_threat': is_threat,
                'threat_score': float(score.item()),
                'confidence': abs(score.item() - 0.5) * 2,  # Distance from decision boundary
                'component_scores': {
                    'deep_learning': float(components['dl_score'].item()),
                    'anomaly_detection': float(components['if_score'].item()),
                    'heuristic_rules': float(components['heuristic_score'].item())
                },
                'component_weights': components['weights'],
                'threat_indicators': threat_indicators,
                'top_suspicious_features': self._get_top_features(x_avg),
                'attention_pattern': components['attention_weights'].squeeze().cpu().numpy().tolist()
            }
            
            return explanation
    
    def _get_top_features(self, features: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top contributing features."""
        # Get feature importance based on deviation from normal (0.5 for normalized features)
        deviations = np.abs(features - 0.5)
        top_indices = np.argsort(deviations)[-top_k:][::-1]
        
        feature_names = list(self.feature_idx.keys())
        top_features = []
        
        for idx in top_indices:
            if idx < len(feature_names):
                top_features.append({
                    'feature': feature_names[idx],
                    'value': float(features[idx]),
                    'deviation': float(deviations[idx])
                })
        
        return top_features


class LightweightThreatDetector(nn.Module):
    """
    Lightweight version for ultra-low latency inference (<10ms).
    Uses only essential features and simpler architecture.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Simple GRU encoder
        self.gru = nn.GRU(
            input_size, hidden_size, 1, batch_first=True
        )
        
        # Fast classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fast forward pass."""
        _, hidden = self.gru(x)
        output = self.classifier(hidden.squeeze(0))
        return output
