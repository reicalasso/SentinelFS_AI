"""
Ensemble training and management for model robustness and performance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from ..models.advanced_models import EnsembleAnalyzer
from ..training.trainer import train_model
from ..training.metrics import calculate_metrics, evaluate_model
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleManager:
    """
    Manages ensemble of models for improved performance and robustness.
    """
    
    def __init__(
        self,
        input_size: int,
        ensemble_size: int = 5,
        base_architecture: str = 'lstm',  # 'lstm', 'transformer', 'cnn-lstm', 'mixed'
        hidden_size: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        seq_len: int = 20
    ):
        self.input_size = input_size
        self.ensemble_size = ensemble_size
        self.base_architecture = base_architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len
        
        # Initialize ensemble models
        self.models = self._initialize_ensemble()
        self.weights = [1.0] * ensemble_size  # Equal initial weights
        
    def _initialize_ensemble(self) -> List[nn.Module]:
        """Initialize the ensemble of models."""
        models = []
        
        if self.base_architecture == 'mixed':
            # Create diverse models for diversity
            from ..models.behavioral_analyzer import BehavioralAnalyzer
            from ..models.advanced_models import TransformerBehavioralAnalyzer, CNNLSTMAnalyzer
            
            for i in range(self.ensemble_size):
                if i % 3 == 0:
                    # LSTM-based model
                    model = BehavioralAnalyzer(
                        input_size=self.input_size,
                        hidden_size=self.hidden_size,
                        num_layers=self.num_layers,
                        dropout=self.dropout,
                        use_attention=True,
                        bidirectional=True
                    )
                elif i % 3 == 1:
                    # Transformer-based model
                    model = TransformerBehavioralAnalyzer(
                        input_size=self.input_size,
                        d_model=self.hidden_size,
                        nhead=8,
                        num_layers=self.num_layers,
                        dropout=self.dropout,
                        seq_len=self.seq_len
                    )
                else:
                    # CNN-LSTM model
                    model = CNNLSTMAnalyzer(
                        input_size=self.input_size,
                        hidden_size=self.hidden_size,
                        num_layers=self.num_layers,
                        dropout=self.dropout
                    )
                models.append(model)
        else:
            # Create models of the same architecture with different random initializations
            for i in range(self.ensemble_size):
                if self.base_architecture == 'lstm':
                    from ..models.behavioral_analyzer import BehavioralAnalyzer
                    model = BehavioralAnalyzer(
                        input_size=self.input_size,
                        hidden_size=self.hidden_size,
                        num_layers=self.num_layers,
                        dropout=self.dropout,
                        use_attention=True,
                        bidirectional=True
                    )
                elif self.base_architecture == 'transformer':
                    from ..models.advanced_models import TransformerBehavioralAnalyzer
                    model = TransformerBehavioralAnalyzer(
                        input_size=self.input_size,
                        d_model=self.hidden_size,
                        nhead=8,
                        num_layers=self.num_layers,
                        dropout=self.dropout,
                        seq_len=self.seq_len
                    )
                elif self.base_architecture == 'cnn-lstm':
                    from ..models.advanced_models import CNNLSTMAnalyzer
                    model = CNNLSTMAnalyzer(
                        input_size=self.input_size,
                        hidden_size=self.hidden_size,
                        num_layers=self.num_layers,
                        dropout=self.dropout
                    )
                else:
                    raise ValueError(f"Unknown architecture: {self.base_architecture}")
                
                models.append(model)
        
        return models
    
    def train_ensemble(
        self,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        epochs: int = 50,
        lr: float = 0.001,
        device: Optional[torch.device] = None
    ) -> List[Dict[str, List[float]]]:
        """
        Train all models in the ensemble.
        
        Args:
            dataloaders: Training and validation dataloaders
            epochs: Number of training epochs
            lr: Learning rate
            device: Training device
            
        Returns:
            List of training histories for each model
        """
        histories = []
        
        for i, model in enumerate(self.models):
            logger.info(f"Training ensemble model {i+1}/{self.ensemble_size}")
            
            # Train the model
            history = train_model(
                model=model,
                dataloaders=dataloaders,
                epochs=epochs,
                lr=lr,
                patience=10,  # Early stopping patience
                device=device
            )
            
            histories.append(history)
            
        return histories
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions using ensemble averaging.
        
        Args:
            x: Input tensor
            
        Returns:
            (ensemble_predictions, individual_predictions) tuple
        """
        device = next(self.models[0].parameters()).device
        x = x.to(device)
        
        individual_preds = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                pred = model(x)
                individual_preds.append(pred)
        
        # Stack individual predictions
        individual_tensor = torch.stack(individual_preds, dim=1)  # (batch, ensemble_size, 1)
        
        # Calculate weighted average
        weights_tensor = torch.tensor(self.weights, device=device).view(1, -1, 1)  # (1, ensemble_size, 1)
        weighted_preds = individual_tensor * weights_tensor
        total_weight = torch.sum(weights_tensor, dim=1, keepdim=True)  # (1, 1, 1)
        ensemble_pred = torch.sum(weighted_preds, dim=1, keepdim=True) / total_weight  # (batch, 1, 1)
        ensemble_pred = ensemble_pred.squeeze(-1)  # (batch, 1)
        
        return ensemble_pred, individual_tensor
    
    def evaluate_ensemble(
        self,
        test_loader: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None
    ) -> Dict[str, float]:
        """
        Evaluate the ensemble performance.
        
        Args:
            test_loader: Test data loader
            device: Evaluation device
            
        Returns:
            Dictionary with evaluation metrics
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        all_ensemble_preds = []
        all_individual_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                ensemble_pred, individual_preds = self.predict(batch_data)
                
                # Ensure we flatten the predictions properly
                pred_np = ensemble_pred.cpu().numpy()
                if pred_np.ndim > 1:
                    pred_np = pred_np.flatten()
                all_ensemble_preds.extend(pred_np)
                
                all_individual_preds.append(individual_preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        all_ensemble_preds = np.array(all_ensemble_preds)
        all_labels = np.array(all_labels).flatten()
        
        # Calculate ensemble metrics
        ensemble_metrics = calculate_metrics(all_ensemble_preds, all_labels)
        
        # Calculate diversity metrics
        # Concatenate all individual predictions along batch dimension
        all_individual_preds = np.concatenate(all_individual_preds, axis=0)  # (total_batch, ensemble_size, 1)
        diversity = self._calculate_diversity(all_individual_preds)
        
        ensemble_metrics['diversity'] = diversity
        
        return ensemble_metrics
    
    def _calculate_diversity(self, individual_preds: np.ndarray) -> float:
        """Calculate diversity of ensemble predictions."""
        # Measure how much predictions vary across ensemble members
        # Higher variance means higher diversity
        pred_std = np.std(individual_preds, axis=1)  # Std across ensemble members
        avg_std = np.mean(pred_std)
        return float(avg_std)
    
    def update_weights(
        self,
        validation_loader: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None
    ):
        """
        Update ensemble weights based on validation performance.
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_scores = []
        
        for model in self.models:
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_data, batch_labels in validation_loader:
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    pred = model(batch_data)
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())
            
            all_preds = np.array(all_preds).flatten()
            all_labels = np.array(all_labels).flatten()
            
            # Calculate F1 score as performance metric
            metrics = calculate_metrics(all_preds, all_labels)
            model_scores.append(metrics['f1_score'])
        
        # Update weights based on performance
        # Better performing models get higher weights
        max_score = max(model_scores)
        if max_score > 0:
            self.weights = [score / max_score for score in model_scores]
        else:
            self.weights = [1.0] * len(model_scores)  # Equal weights if all bad
        
        logger.info(f"Updated ensemble weights: {self.weights}")
    
    def save_ensemble(self, save_dir: Path):
        """Save all ensemble models."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for i, model in enumerate(self.models):
            model_path = save_dir / f"model_{i}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': type(model).__name__,
                'architecture': self.base_architecture,
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'seq_len': self.seq_len
            }, model_path)
        
        # Save ensemble configuration
        config = {
            'ensemble_size': self.ensemble_size,
            'base_architecture': self.base_architecture,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'seq_len': self.seq_len,
            'weights': self.weights
        }
        
        with open(save_dir / 'ensemble_config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_ensemble(self, save_dir: Path):
        """Load all ensemble models."""
        save_dir = Path(save_dir)
        
        # Load ensemble configuration
        with open(save_dir / 'ensemble_config.json', 'r') as f:
            config = json.load(f)
        
        # Update ensemble parameters
        self.ensemble_size = config['ensemble_size']
        self.base_architecture = config['base_architecture']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.seq_len = config['seq_len']
        self.weights = config['weights']
        
        # Reinitialize models
        self.models = self._initialize_ensemble()
        
        # Load each model
        for i, model in enumerate(self.models):
            model_path = save_dir / f"model_{i}.pt"
            checkpoint = torch.load(model_path, weights_only=False)
            
            model.load_state_dict(checkpoint['model_state_dict'])


def create_weighted_ensemble(
    models: List[nn.Module],
    weights: Optional[List[float]] = None
) -> EnsembleAnalyzer:
    """
    Create a weighted ensemble from pre-trained models.
    
    Args:
        models: List of pre-trained models
        weights: Optional weights for each model (default: equal weights)
        
    Returns:
        EnsembleAnalyzer model
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    # Assuming all models have the same input size
    input_size = models[0].input_size if hasattr(models[0], 'input_size') else 7
    
    ensemble = EnsembleAnalyzer(
        input_size=input_size,
        hidden_size=128,
        num_layers=3,
        dropout=0.3,
        seq_len=20,
        weights=weights
    )
    
    # Note: This is a simplified ensemble that uses pre-trained models
    # without retraining the fusion layer
    ensemble.lstm_model = models[0].lstm if hasattr(models[0], 'lstm') else models[0]
    ensemble.transformer_model = models[1] if len(models) > 1 else models[0]
    ensemble.cnn_lstm_model = models[2] if len(models) > 2 else models[0]
    
    return ensemble