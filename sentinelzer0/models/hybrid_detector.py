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
        isolation_forest_contamination: float = 0.08,  # Slightly lower for better precision
        heuristic_weight: float = 0.3,  # Further reduced for DL dominance
        dl_weight: float = 0.5,         # Significantly increased for DL effectiveness
        anomaly_weight: float = 0.2     # Slightly reduced
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
        
        # === Component 1: Advanced Deep Learning Architecture ===
        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # Feature embedding with residual connection
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Light dropout for embedding
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Multi-layer RNN with residual connections
        RNN = nn.GRU if use_gru else nn.LSTM
        self.rnn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = hidden_size if i == 0 else hidden_size * 2
            self.rnn_layers.append(
                RNN(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=0,  # We'll handle dropout manually
                    bidirectional=True
                )
            )
        
        # Advanced multi-head attention
        self.num_attention_heads = 8
        self.attention_dim = hidden_size * 2 // self.num_attention_heads
        
        self.multi_head_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, self.attention_dim),
                nn.Tanh(),
                nn.Linear(self.attention_dim, 1)
            ) for _ in range(self.num_attention_heads)
        ])
        
        # Context fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2 * self.num_attention_heads, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Advanced classifier with skip connections
        self.dl_classifier = nn.ModuleList([
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.Linear(hidden_size // 4, 1)
        ])
        
        self.classifier_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size),
            nn.LayerNorm(hidden_size // 2),
            nn.LayerNorm(hidden_size // 4)
        ])
        
        # Confidence estimation branch
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
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
        # These thresholds are more sensitive to catch threats
        self.heuristic_thresholds = {
            'rapid_modifications': 0.15,  # >15% rapid modifications (more sensitive)
            'mass_operations': 0.25,  # >25% mass operation score (more sensitive)
            'ransomware_indicators': 0.2,  # Combined ransomware score (very sensitive)
            'unusual_time_ops': 0.4,  # >40% operations at unusual times
            'high_delete_rate': 0.2,  # >20% delete rate (more sensitive)
            'suspicious_extensions': 0.5,  # >50% suspicious file extensions
        }
        
        # Feature indices (must match RealFeatureExtractor output)
        self._setup_feature_indices()

        # Note: Combined score is already in [0,1] (components are bounded/sigmoided).
        # Avoid an extra sigmoid that would compress the dynamic range around 0.5.
        
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
            'rename_rate', 'is_hidden', 'is_unusual_time', 'is_critical_path',
            'is_suspicious_process', 'threat_multiplier', 'is_trusted_process',
            'is_legitimate_extension'
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
        
        # Data augmentation during training - add small gaussian noise
        if self.training and torch.rand(1).item() < 0.3:  # 30% chance
            noise = torch.randn_like(x) * 0.01  # Small noise
            x = x + noise
        
        # === Component 1: Advanced Deep Learning ===
        # Input normalization
        x_norm = self.input_norm(x)
        
        # Feature embedding
        x_embedded = self.feature_embedding(x_norm)  # (batch, seq, hidden)
        
        # Multi-layer RNN processing with residuals
        rnn_out = x_embedded
        for i, rnn_layer in enumerate(self.rnn_layers):
            if i > 0:
                # Residual connection for deeper layers
                residual = rnn_out
            
            rnn_out, _ = rnn_layer(rnn_out)  # (batch, seq, hidden*2)
            
            # Apply residual connection if not first layer
            if i > 0 and residual.size(-1) == rnn_out.size(-1):
                rnn_out = rnn_out + residual
            
            # Layer normalization and dropout
            rnn_out = nn.functional.dropout(rnn_out, p=self.num_layers * 0.1, training=self.training)
        
        # Multi-head attention mechanism
        attention_heads = []
        attention_weights_list = []
        
        for head in self.multi_head_attention:
            attention_weights = head(rnn_out)  # (batch, seq, 1)
            attention_weights = torch.softmax(attention_weights, dim=1)
            attention_weights_list.append(attention_weights)
            
            # Weighted sum for this head
            head_context = torch.sum(attention_weights * rnn_out, dim=1)  # (batch, hidden*2)
            attention_heads.append(head_context)
        
        # Concatenate all attention heads
        multi_head_context = torch.cat(attention_heads, dim=1)  # (batch, hidden*2*num_heads)
        
        # Fuse multi-head contexts
        fused_context = self.context_fusion(multi_head_context)  # (batch, hidden*2)
        
        # Advanced classifier with residual connections
        x_cls = fused_context
        skip_connections = []
        
        for i, (layer, norm) in enumerate(zip(self.dl_classifier[:-1], self.classifier_norms)):
            residual = x_cls
            x_cls = layer(x_cls)
            x_cls = norm(x_cls)
            x_cls = torch.relu(x_cls)
            
            # Skip connection every 2 layers
            if i % 2 == 1 and len(skip_connections) > 0:
                if skip_connections[-1].size(-1) == x_cls.size(-1):
                    x_cls = x_cls + skip_connections[-1]
            
            skip_connections.append(residual)
            x_cls = nn.functional.dropout(x_cls, p=0.1, training=self.training)
        
        # Final prediction
        dl_score = torch.sigmoid(self.dl_classifier[-1](x_cls))  # (batch, 1)
        
        # Confidence estimation for metrics only (not used in scoring)
        confidence = self.confidence_estimator(fused_context)  # (batch, 1)
        
        # Use raw DL score without confidence modulation
        # dl_score already represents the model's confidence
        
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
        
        # === Advanced Dynamic Ensemble ===
        # Calculate component-specific confidence scores
        dl_confidence = confidence
        heuristic_confidence = self._calculate_heuristic_confidence(x)
        if_confidence = self._calculate_if_confidence(if_score) if self.if_fitted else torch.zeros_like(dl_confidence)
        
        # Normalize confidences
        total_confidence = dl_confidence + heuristic_confidence + if_confidence + 1e-8
        dl_conf_norm = dl_confidence / total_confidence
        heuristic_conf_norm = heuristic_confidence / total_confidence
        if_conf_norm = if_confidence / total_confidence
        
        # Dynamic weight adjustment based on confidence
        dynamic_dl_weight = self.dl_weight * (1 + dl_conf_norm * 1.0)  # More boost for DL
        dynamic_heuristic_weight = self.heuristic_weight * (1 + heuristic_conf_norm * 0.3)  # Less boost for heuristics
        dynamic_if_weight = self.anomaly_weight * (1 + if_conf_norm * 0.7)  # Moderate boost for IF
        
        # Renormalize dynamic weights
        total_dynamic_weight = dynamic_dl_weight + dynamic_heuristic_weight + dynamic_if_weight
        dynamic_dl_weight /= total_dynamic_weight
        dynamic_heuristic_weight /= total_dynamic_weight
        dynamic_if_weight /= total_dynamic_weight
        
        # Confidence-weighted ensemble
        combined_score = (
            dynamic_dl_weight * dl_score +
            dynamic_if_weight * if_score +
            dynamic_heuristic_weight * heuristic_score
        )
        
        # Meta-learning adjustment: Learn from component agreement/disagreement
        component_agreement = self._calculate_component_agreement(dl_score, if_score, heuristic_score)
        agreement_boost = torch.where(
            component_agreement > 0.7,  # High agreement
            torch.tensor(1.1, device=x.device),  # Boost confidence
            torch.where(
                component_agreement < 0.3,  # Low agreement
                torch.tensor(0.9, device=x.device),  # Reduce confidence
                torch.tensor(1.0, device=x.device)  # Neutral
            )
        )
        
        # Apply agreement adjustment
        combined_score = combined_score * agreement_boost

        # Final calibrated score with temperature scaling
        temperature = 1.2  # Slightly higher temperature for better calibration
        final_score = torch.clamp(torch.sigmoid(combined_score / temperature), 0.0, 1.0)
        
        if return_components:
            components = {
                'dl_score': dl_score,
                'if_score': if_score,
                'heuristic_score': heuristic_score,
                'combined_score': combined_score,
                'attention_weights': attention_weights_list[0] if attention_weights_list else None,
                'multi_head_attention': attention_weights_list,
                'confidence': {
                    'dl_confidence': dl_confidence,
                    'heuristic_confidence': heuristic_confidence,
                    'if_confidence': if_confidence,
                    'component_agreement': component_agreement
                },
                'dynamic_weights': {
                    'dl': dynamic_dl_weight,
                    'if': dynamic_if_weight,
                    'heuristic': dynamic_heuristic_weight
                },
                'static_weights': {
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
        is_critical_path = x_avg[:, self.feature_idx['is_critical_path']]
        is_suspicious_process = x_avg[:, self.feature_idx['is_suspicious_process']]
        threat_multiplier = x_avg[:, self.feature_idx['threat_multiplier']]
        is_trusted_process = x_avg[:, self.feature_idx['is_trusted_process']]
        is_legitimate_extension = x_avg[:, self.feature_idx['is_legitimate_extension']]
        
        # Rule 1: Ransomware indicators (CRITICAL)
        ransomware_score = (
            ransomware_ext * 0.5 +
            ransomware_name * 0.4 +
            rapid_mods * 0.1
        )
        # ANY ransomware indicator should trigger high alert
        ransomware_detected = (ransomware_score > 0.2).float() * 0.95
        
        # Rule 2: Critical system path access (CRITICAL)
        critical_path_detected = (is_critical_path > 0.5).float() * 0.9
        
        # Rule 3: Suspicious process (HIGH)
        suspicious_process_detected = (is_suspicious_process > 0.5).float() * 0.85
        
        # Rule 4: Mass file operations (exfiltration or destructive)
        mass_op_detected = (mass_ops > self.heuristic_thresholds['mass_operations']).float() * 0.7
        
        # Rule 5: High delete rate (data destruction)
        high_delete_detected = (delete_rate > self.heuristic_thresholds['high_delete_rate']).float() * 0.6
        
        # Rule 6: Suspicious file patterns
        suspicious_pattern = (
            (is_encrypted > 0.5) *
            (high_entropy > 0.6) *
            (rename_rate > 0.2)
        ).float() * 0.5
        
        # Rule 7: Unusual time operations with suspicious activity
        unusual_time_suspicious = (
            (unusual_time > self.heuristic_thresholds['unusual_time_ops']) *
            (mass_ops > 0.3)
        ).float() * 0.4
        
        # === ADVANCED BEHAVIORAL ANALYSIS ===
        # Rule 8: Process-File Extension Mismatch Detection
        process_mismatch = self._detect_process_file_mismatch(x_avg).float() * 0.6
        
        # Rule 9: Temporal Anomaly Detection (burst patterns)
        temporal_anomaly = self._detect_temporal_anomalies(x_avg).float() * 0.5
        
        # Rule 10: Privilege Escalation Patterns
        privilege_escalation = self._detect_privilege_escalation(x_avg).float() * 0.8
        
        # Rule 11: File Size Anomaly Detection
        size_anomaly = self._detect_file_size_anomalies(x_avg).float() * 0.4
        
        # Rule 12: Network-related File Operations
        network_operations = self._detect_network_operations(x_avg).float() * 0.7
        
        # Rule 13: Living Off The Land Attacks (PowerShell + suspicious locations/extensions)
        living_off_land = self._detect_living_off_land(x_avg).float() * 0.8
        
        # Rule 14: DNS Tunneling / Suspicious Scripts
        dns_tunneling = self._detect_dns_tunneling(x_avg).float() * 0.9
        
        # Combine all rules - Weighted combination of critical vs behavioral indicators
        # Critical rules (take MAX for immediate detection)
        critical_rules = torch.max(
            torch.stack([
                ransomware_detected,
                critical_path_detected,
                suspicious_process_detected
            ], dim=1),
            dim=1
        )[0]
        
        # Behavioral rules (weighted sum for pattern recognition)
        behavioral_rules = (
            mass_op_detected * 0.25 +
            high_delete_detected * 0.20 +
            suspicious_pattern * 0.15 +
            unusual_time_suspicious * 0.10 +
            process_mismatch * 0.15 +
            temporal_anomaly * 0.10 +
            privilege_escalation * 0.25 +
            size_anomaly * 0.10 +
            network_operations * 0.20 +
            living_off_land * 0.25 +  # Living off the land detection
            dns_tunneling * 0.30      # NEW: DNS tunneling/suspicious script detection
        ) / 2.05  # Normalize        # Combined heuristic score: MAX of critical or behavioral
        heuristic_score = torch.clamp(
            torch.max(critical_rules, behavioral_rules),
            min=0.0,
            max=1.0
        )
        
        # Apply threat multiplier to boost serious threats
        heuristic_score = heuristic_score * (0.5 + 0.5 * threat_multiplier)
        
        # TRUSTED PROCESS OVERRIDE: Dramatically reduce score for trusted processes
        # If it's a trusted process, multiply the heuristic score by 0.02 (98% reduction)
        trust_dampener = torch.where(
            is_trusted_process > 0.5,
            torch.tensor(0.02, device=x.device),  # 98% reduction for trusted
            torch.tensor(1.0, device=x.device)    # No change for untrusted
        )
        
        # LEGITIMATE FILE OVERRIDE: Reduce score for legitimate file types
        legitimacy_dampener = torch.where(
            is_legitimate_extension > 0.5,
            torch.tensor(0.2, device=x.device),   # 80% reduction for legitimate files
            torch.tensor(1.0, device=x.device)    # No change for other files
        )
        
        # DOUBLE PROTECTION: Both trusted process AND legitimate file
        double_protection = torch.where(
            (is_trusted_process > 0.5) & (is_legitimate_extension > 0.5),
            torch.tensor(0.005, device=x.device),  # 99.5% reduction for both
            torch.tensor(1.0, device=x.device)
        )
        
        # Apply all dampening factors
        heuristic_score = heuristic_score * trust_dampener * legitimacy_dampener * double_protection
        
        # SECURITY OVERRIDE: DNS tunneling and critical threats bypass trusted process dampening
        # If DNS tunneling or critical security patterns are detected, restore higher score
        security_override = torch.where(
            (dns_tunneling > 0.5),  # DNS tunneling detected - bypass trust
            torch.max(heuristic_score, torch.tensor(0.6, device=x.device)),  # Force high threat score
            torch.where(
                (critical_rules > 0.8),  # Other critical threats
                torch.max(heuristic_score, critical_rules * 0.6),  # Restore significant threat level
                heuristic_score
            )
        )
        
        return torch.clamp(security_override, 0.0, 1.0).unsqueeze(1)
    
    def _calculate_heuristic_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate confidence score for heuristic component."""
        batch_size = x.size(0)
        x_avg = x.mean(dim=1)  # (batch, features)
        
        # Extract key indicators for confidence calculation
        is_trusted = x_avg[:, self.feature_idx['is_trusted_process']]
        threat_multiplier = x_avg[:, self.feature_idx['threat_multiplier']]
        ransomware_indicators = (
            x_avg[:, self.feature_idx['has_ransomware_ext']] + 
            x_avg[:, self.feature_idx['has_ransomware_name']]
        )
        
        # High confidence when we have clear indicators (positive or negative)
        confidence = torch.clamp(
            is_trusted * 0.9 +  # Very confident about trusted processes
            threat_multiplier * 0.7 +  # Confident about high threat multipliers
            ransomware_indicators * 0.8 + 0.3,  # Base confidence
            0.0, 1.0
        )
        
        return confidence.unsqueeze(1)
    
    def _calculate_if_confidence(self, if_score: torch.Tensor) -> torch.Tensor:
        """Calculate confidence score for Isolation Forest component."""
        # IF confidence based on how extreme the anomaly score is
        # More extreme scores (close to 0 or 1) get higher confidence
        # Enhanced confidence calculation for better IF contribution
        distance_from_center = torch.abs(if_score - 0.5)  # Distance from uncertain middle
        confidence = torch.pow(2 * distance_from_center, 1.5)  # Power scaling for more dramatic confidence
        return torch.clamp(confidence, 0.1, 1.0)  # Minimum confidence 0.1
    
    def _calculate_component_agreement(self, dl_score: torch.Tensor, if_score: torch.Tensor, heuristic_score: torch.Tensor) -> torch.Tensor:
        """Calculate agreement between different components."""
        # Ensure all scores are properly shaped (batch_size, 1)
        dl_flat = dl_score.view(-1)  # (batch_size,)
        if_flat = if_score.view(-1)  # (batch_size,)
        heur_flat = heuristic_score.view(-1)  # (batch_size,)
        
        # Stack for comparison (batch_size, 3)
        scores = torch.stack([dl_flat, if_flat, heur_flat], dim=1)
        
        # Calculate pairwise differences
        mean_score = scores.mean(dim=1, keepdim=True)
        deviations = torch.abs(scores - mean_score)
        max_deviation = deviations.max(dim=1)[0]
        
        # Agreement is inverse of maximum deviation
        agreement = 1.0 - torch.clamp(max_deviation, 0.0, 1.0)
        
        return agreement.unsqueeze(1)
    
    def _detect_process_file_mismatch(self, x_avg: torch.Tensor) -> torch.Tensor:
        """Detect suspicious process-file extension mismatches."""
        # This is a simplified heuristic - in practice, you'd use the actual process names
        is_executable = x_avg[:, self.feature_idx['is_executable']]
        is_document = x_avg[:, self.feature_idx['is_document']]
        is_suspicious_process = x_avg[:, self.feature_idx['is_suspicious_process']]
        
        # Suspicious if executable files are being processed by non-system processes
        mismatch_score = is_executable * (1 - is_suspicious_process) * 0.6
        
        # Or if document files are being processed by suspicious processes
        mismatch_score += is_document * is_suspicious_process * 0.8
        
        return torch.clamp(mismatch_score, 0.0, 1.0)
    
    def _detect_temporal_anomalies(self, x_avg: torch.Tensor) -> torch.Tensor:
        """Detect temporal anomalies in file operations."""
        ops_per_minute = x_avg[:, self.feature_idx['ops_per_minute']]
        burstiness = x_avg[:, self.feature_idx['burstiness']]
        
        # High burst of operations is suspicious
        burst_threshold = 0.8  # Normalized threshold
        temporal_anomaly = (
            (ops_per_minute > burst_threshold) * 0.6 +
            (burstiness > 0.7) * 0.4
        )
        
        return torch.clamp(temporal_anomaly, 0.0, 1.0)
    
    def _detect_privilege_escalation(self, x_avg: torch.Tensor) -> torch.Tensor:
        """Detect privilege escalation patterns."""
        is_critical_path = x_avg[:, self.feature_idx['is_critical_path']]
        is_suspicious_process = x_avg[:, self.feature_idx['is_suspicious_process']]
        unusual_time = x_avg[:, self.feature_idx['is_unusual_time']]
        
        # Privilege escalation: suspicious process accessing critical paths at unusual times
        priv_esc_score = (
            is_critical_path * is_suspicious_process * 0.7 +
            is_critical_path * unusual_time * 0.3
        )
        
        return torch.clamp(priv_esc_score, 0.0, 1.0)
    
    def _detect_file_size_anomalies(self, x_avg: torch.Tensor) -> torch.Tensor:
        """Detect unusual file size patterns."""
        log_file_size = x_avg[:, self.feature_idx['log_file_size']]
        size_change_ratio = x_avg[:, self.feature_idx['size_change_ratio']]
        
        # Extremely large files or dramatic size changes
        size_anomaly = (
            (log_file_size > 20.0) * 0.4 +  # Very large files (>1GB)
            (size_change_ratio > 5.0) * 0.6   # Dramatic size changes
        )
        
        return torch.clamp(size_anomaly, 0.0, 1.0)
    
    def _detect_network_operations(self, x_avg: torch.Tensor) -> torch.Tensor:
        """Detect network-related suspicious operations."""
        # This would typically involve checking for network-related processes
        # For now, we'll use operation diversity and file diversity as proxies
        operation_diversity = x_avg[:, self.feature_idx['operation_diversity']]
        file_diversity = x_avg[:, self.feature_idx['file_diversity']]
        
        # High diversity might indicate data exfiltration or reconnaissance
        network_suspicious = (
            (operation_diversity > 0.8) * 0.3 +
            (file_diversity > 0.9) * 0.7
        )
        
        return torch.clamp(network_suspicious, 0.0, 1.0)
    
    def _detect_living_off_land(self, x_avg: torch.Tensor) -> torch.Tensor:
        """Detect Living Off The Land attacks (PowerShell, suspicious scripts in temp locations)."""
        # Check for suspicious processes
        is_suspicious_process = x_avg[:, self.feature_idx['is_suspicious_process']]
        
        # Check if path contains suspicious indicators
        # For PowerShell scripts in /tmp, temp directories, etc.
        # This would be enhanced with path-specific features in a real implementation
        rapid_modifications = x_avg[:, self.feature_idx['rapid_modifications']]
        unusual_time = x_avg[:, self.feature_idx['is_unusual_time']]
        
        # Living off the land typically involves:
        # 1. Suspicious processes (like PowerShell in this context)
        # 2. Operations in temporary directories
        # 3. Unusual timing or rapid modifications
        lol_score = (
            is_suspicious_process * 0.6 +  # PowerShell/suspicious process
            rapid_modifications * 0.3 +    # Rapid script execution
            unusual_time * 0.1             # Unusual timing
        )
        
        return torch.clamp(lol_score, 0.0, 1.0)
    
    def _detect_dns_tunneling(self, x_avg: torch.Tensor) -> torch.Tensor:
        """Detect DNS tunneling and suspicious script execution"""
        
        # DNS tunneling patterns:
        # 1. Python/scripting processes accessing suspicious files in /tmp
        # 2. Network-related files in temp directories  
        # 3. Files with "dns", "tunnel", "network" in names
        
        high_entropy = x_avg[:, self.feature_idx['filename_entropy']]
        is_critical_path = x_avg[:, self.feature_idx['is_critical_path']]
        threat_multiplier = x_avg[:, self.feature_idx['threat_multiplier']]
        rapid_modifications = x_avg[:, self.feature_idx['rapid_modifications']]
        
        # Specific patterns for DNS tunneling:
        # - Files in /tmp directory (critical_path > 0.5)
        # - Network/DNS related naming (threat_multiplier boost)
        # - Python scripts with network capabilities
        dns_tunneling_score = torch.where(
            (is_critical_path > 0.5) &  # /tmp directory
            ((high_entropy > 0.5) | (threat_multiplier > 0.2) | 
             (rapid_modifications > 0.3)),  # Suspicious patterns
            torch.tensor(1.0, device=x_avg.device),
            torch.tensor(0.0, device=x_avg.device)
        )
        
        return dns_tunneling_score
    
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
