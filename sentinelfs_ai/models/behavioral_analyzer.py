"""
Advanced LSTM-based behavioral analyzer with attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .attention import AttentionLayer


class BehavioralAnalyzer(nn.Module):
    """
    Advanced LSTM-based behavioral analyzer with attention mechanism.
    
    This model implements the AI component described in the academic paper,
    with enhanced architecture including:
    - Multi-layer LSTM with dropout
    - Self-attention mechanism for temporal patterns
    - Layer normalization
    - Residual connections
    - Multi-head classification
    
    Architecture:
        - Multi-layer Bidirectional LSTM with dropout
        - Self-attention layer
        - Layer normalization
        - Fully connected classifier with residual connection
        - Sigmoid activation for binary classification
    
    Args:
        input_size: Number of features per timestep
        hidden_size: Number of LSTM hidden units (default: 64)
        num_layers: Number of LSTM layers (default: 3)
        dropout: Dropout probability (default: 0.3)
        use_attention: Whether to use attention mechanism (default: True)
        bidirectional: Whether to use bidirectional LSTM (default: True)
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 3, dropout: float = 0.3,
                 use_attention: bool = True, bidirectional: bool = True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        
        # Calculate actual hidden size for bidirectional LSTM
        lstm_hidden = hidden_size // 2 if bidirectional else hidden_size
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size, 
            lstm_hidden, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionLayer(hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Feature transformation for residual connection
        self.feature_transform = nn.Linear(input_size, hidden_size)
        
        # Enhanced classifier with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Auxiliary classifier for multi-task learning
        self.aux_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 4)  # 4 anomaly types
        )
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            return_attention: Whether to return attention weights
            
        Returns:
            Anomaly scores of shape (batch_size, 1) in range [0, 1]
            If return_attention is True, also returns attention weights
        """
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention or use last output
        if self.use_attention:
            context_vector, attention_weights = self.attention(lstm_out)
        else:
            context_vector = lstm_out[:, -1, :]
            attention_weights = None
        
        # Apply layer normalization
        normalized = self.layer_norm(context_vector)
        
        # Residual connection from input
        input_features = self.feature_transform(x.mean(dim=1))  # Average pooling over time
        normalized = normalized + 0.3 * input_features  # Scaled residual
        
        # Apply dropout
        dropped = self.dropout(normalized)
        
        # Main classification
        output = self.classifier(dropped)
        
        # Apply sigmoid for probability
        output = torch.sigmoid(output)
        
        if return_attention and attention_weights is not None:
            return output, attention_weights
        return output
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract LSTM embeddings for analysis.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Embeddings of shape (batch_size, hidden_size)
        """
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            if self.use_attention:
                context_vector, _ = self.attention(lstm_out)
                return context_vector
            return lstm_out[:, -1, :]
    
    def predict_anomaly_type(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the type of anomaly (multi-class classification).
        
        Types:
            0: Normal
            1: Data Exfiltration
            2: Ransomware
            3: Privilege Escalation
            4: Other Anomaly
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Class probabilities of shape (batch_size, 4)
        """
        with torch.no_grad():
            embeddings = self.get_embeddings(x)
            return F.softmax(self.aux_classifier(embeddings), dim=1)
