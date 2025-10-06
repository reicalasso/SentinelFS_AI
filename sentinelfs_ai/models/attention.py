"""
Self-attention mechanism for temporal pattern detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AttentionLayer(nn.Module):
    """
    Self-attention mechanism for temporal pattern detection.
    Helps the model focus on important time steps in access sequences.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism.
        
        Args:
            lstm_output: LSTM output of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)
        
        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden)
        
        return context_vector, attention_weights.squeeze(-1)
