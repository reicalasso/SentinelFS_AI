"""
Diverse Model Architectures for Ensemble

Implements multiple architectures to promote ensemble diversity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging


class CNNDetector(nn.Module):
    """
    CNN-based malware detector.
    
    Uses convolutional layers for feature extraction.
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_channels: int = 32,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize CNN detector.
        
        Args:
            input_dim: Input feature dimension
            hidden_channels: Number of CNN channels
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Reshape input to work with Conv1d
        # Input: (batch, features) -> (batch, 1, features)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.conv3 = nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_channels * 4)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_channels * 4, hidden_channels * 2)
        self.fc2 = nn.Linear(hidden_channels * 2, num_classes)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized CNN detector: {input_dim} -> {num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, features)
        
        Returns:
            Logits (batch, num_classes)
        """
        # Reshape for Conv1d
        x = x.unsqueeze(1)  # (batch, 1, features)
        
        # Conv blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.global_pool(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)
        
        # Classifier
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class LSTMDetector(nn.Module):
    """
    LSTM-based malware detector.
    
    Uses recurrent layers for temporal modeling.
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM detector.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=1,  # Each feature as a sequence step
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Classifier
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized LSTM detector: {input_dim} -> {num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, features)
        
        Returns:
            Logits (batch, num_classes)
        """
        # Reshape for LSTM: treat features as sequence
        x = x.unsqueeze(-1)  # (batch, seq_len=features, input_size=1)
        
        # LSTM
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat([hn[-2], hn[-1]], dim=-1)
        else:
            hidden = hn[-1]
        
        # Classifier
        x = self.dropout(hidden)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class TransformerDetector(nn.Module):
    """
    Transformer-based malware detector.
    
    Uses self-attention for feature relationships.
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize Transformer detector.
        
        Args:
            input_dim: Input feature dimension
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=input_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, num_classes)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized Transformer detector: {input_dim} -> {num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, features)
        
        Returns:
            Logits (batch, num_classes)
        """
        # Reshape: treat each feature as a token
        x = x.unsqueeze(-1)  # (batch, seq_len=features, 1)
        
        # Project to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classifier
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ResidualBlock(nn.Module):
    """Residual block for deeper architectures."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        residual = x
        
        x = self.norm1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        x = x + residual
        x = self.norm2(x)
        
        return x


class DeepMLPDetector(nn.Module):
    """
    Deep MLP with residual connections.
    
    Alternative architecture for ensemble diversity.
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize deep MLP detector.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_blocks: Number of residual blocks
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized Deep MLP detector: {input_dim} -> {num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, features)
        
        Returns:
            Logits (batch, num_classes)
        """
        # Input projection
        x = F.relu(self.input_proj(x))
        x = self.dropout(x)
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.output_proj(x)
        
        return x
