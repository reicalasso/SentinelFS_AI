"""
Data structures and type definitions for SentinelFS AI.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class AnalysisResult:
    """Result of AI behavioral analysis matching API specification."""
    access_pattern_score: float
    behavior_normal: bool
    anomaly_detected: bool
    confidence: float
    last_updated: str
    threat_score: float
    anomaly_type: Optional[str] = None
    anomaly_type_confidence: Optional[float] = None
    attention_weights: Optional[List[float]] = None
    explanation: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
            'access_pattern_score': float(self.access_pattern_score),
            'behavior_normal': self.behavior_normal,
            'anomaly_detected': self.anomaly_detected,
            'confidence': float(self.confidence),
            'last_updated': self.last_updated,
            'threat_score': float(self.threat_score)
        }
        
        # Add optional fields if available
        if self.anomaly_type:
            result['anomaly_type'] = self.anomaly_type
        if self.anomaly_type_confidence:
            result['anomaly_type_confidence'] = float(self.anomaly_type_confidence)
        if self.attention_weights:
            result['attention_weights'] = [float(w) for w in self.attention_weights]
        if self.explanation:
            result['explanation'] = self.explanation
        
        return result


@dataclass
class AnomalyType:
    """Enumeration of anomaly types."""
    NORMAL = 0
    DATA_EXFILTRATION = 1
    RANSOMWARE = 2
    PRIVILEGE_ESCALATION = 3
    OTHER = 4
    
    @staticmethod
    def get_name(type_id: int) -> str:
        """Get human-readable name for anomaly type."""
        names = {
            0: "Normal",
            1: "Data Exfiltration",
            2: "Ransomware",
            3: "Privilege Escalation",
            4: "Other Anomaly"
        }
        return names.get(type_id, "Unknown")
    
    @staticmethod
    def get_description(type_id: int) -> str:
        """Get detailed description for anomaly type."""
        descriptions = {
            0: "Normal access pattern within expected parameters",
            1: "Large file transfers during off-hours suggesting data theft",
            2: "Rapid file modifications/encryptions indicating ransomware",
            3: "Unusual privilege escalation or administrative access",
            4: "Anomalous pattern not matching specific attack types"
        }
        return descriptions.get(type_id, "Unknown anomaly type")


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    num_samples: int = 1000
    seq_len: int = 10
    anomaly_ratio: float = 0.2
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    patience: int = 10
    hidden_size: int = 64
    num_layers: int = 3
    dropout: float = 0.3
    model_dir: Path = Path('./models')
    checkpoint_dir: Path = Path('./checkpoints')
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
