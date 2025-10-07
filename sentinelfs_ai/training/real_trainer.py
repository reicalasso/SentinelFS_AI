"""
Real-world training system with incremental learning and online adaptation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime
import time

from ..models.hybrid_detector import HybridThreatDetector
from ..data.real_feature_extractor import RealFeatureExtractor
from ..utils.logger import get_logger
from ..utils.device import get_device

logger = get_logger(__name__)


class RealWorldTrainer:
    """
    Production training system that learns from actual file system behavior.
    
    Features:
    - Incremental learning from new data
    - Online adaptation to evolving threats
    - Class imbalance handling
    - Model checkpointing and versioning
    - Performance monitoring
    - Early stopping with patience
    """
    
    def __init__(
        self,
        model: HybridThreatDetector,
        feature_extractor: RealFeatureExtractor,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        class_weight_positive: float = 2.0,  # Weight for positive (anomaly) class
        patience: int = 10,
        checkpoint_dir: str = './checkpoints'
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = get_device()
        self.model = self.model.to(self.device)
        
        # Optimizer with L2 regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Loss function with class weights
        self.class_weight_positive = class_weight_positive
        
        # Early stopping
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        # Performance metrics
        self.metrics = {}
    
    def train_from_real_data(
        self,
        train_events: List[Dict[str, Any]],
        val_events: List[Dict[str, Any]],
        train_labels: np.ndarray,
        val_labels: np.ndarray,
        num_epochs: int = 50,
        batch_size: int = 32,
        sequence_length: int = 50
    ) -> Dict[str, Any]:
        """
        Train model from real file system events.
        
        Args:
            train_events: List of training events
            val_events: List of validation events
            train_labels: Training labels (0=normal, 1=anomaly)
            val_labels: Validation labels
            num_epochs: Number of training epochs
            batch_size: Batch size
            sequence_length: Length of event sequences
            
        Returns:
            Training history and metrics
        """
        logger.info(f"Starting training on {len(train_events)} events...")
        logger.info(f"Sequence length: {sequence_length}, Batch size: {batch_size}")
        
        # Extract features from events
        logger.info("Extracting features from training data...")
        train_features = self._prepare_sequences(train_events, sequence_length)
        val_features = self._prepare_sequences(val_events, sequence_length)
        
        # Fit Isolation Forest on training data
        logger.info("Fitting Isolation Forest...")
        self.model.fit_isolation_forest(train_features)
        
        # Calibrate heuristic thresholds
        logger.info("Calibrating heuristic thresholds...")
        train_labels_aligned = self._align_labels(train_labels, len(train_features))
        val_labels_aligned = self._align_labels(val_labels, len(val_features))
        self.model.calibrate_thresholds(train_features, train_labels_aligned)
        
        # Create data loaders
        train_loader = self._create_dataloader(
            train_features, train_labels_aligned, batch_size, shuffle=True
        )
        val_loader = self._create_dataloader(
            val_features, val_labels_aligned, batch_size, shuffle=False
        )
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
            
            # Log progress
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s) - "
                f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
            
            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model from training")
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f}s")
        
        # Save final model components
        self.model.save_components(str(self.checkpoint_dir / 'final'))
        
        return {
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time,
            'final_metrics': val_metrics
        }
    
    def incremental_update(
        self,
        new_events: List[Dict[str, Any]],
        new_labels: np.ndarray,
        num_epochs: int = 5,
        batch_size: int = 32,
        sequence_length: int = 50
    ) -> Dict[str, float]:
        """
        Perform incremental learning on new data.
        
        Args:
            new_events: New file system events
            new_labels: Labels for new events
            num_epochs: Number of update epochs
            batch_size: Batch size
            sequence_length: Sequence length
            
        Returns:
            Update metrics
        """
        logger.info(f"Performing incremental update with {len(new_events)} new events...")
        
        # Extract features
        new_features = self._prepare_sequences(new_events, sequence_length)
        new_labels_aligned = self._align_labels(new_labels, len(new_features))
        
        # Create data loader
        update_loader = self._create_dataloader(
            new_features, new_labels_aligned, batch_size, shuffle=True
        )
        
        # Fine-tune model with lower learning rate
        old_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer.param_groups[0]['lr'] = old_lr * 0.1  # 10x smaller LR
        
        # Update for a few epochs
        total_loss = 0.0
        total_acc = 0.0
        
        for epoch in range(num_epochs):
            metrics = self._train_epoch(update_loader, epoch)
            total_loss += metrics['loss']
            total_acc += metrics['accuracy']
        
        # Restore learning rate
        self.optimizer.param_groups[0]['lr'] = old_lr
        
        # Save updated model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_checkpoint(-1, metrics, is_best=False, suffix=f"_incremental_{timestamp}")
        
        logger.info(f"Incremental update completed - Avg Loss: {total_loss/num_epochs:.4f}")
        
        return {
            'avg_loss': total_loss / num_epochs,
            'avg_accuracy': total_acc / num_epochs,
            'num_samples': len(new_features)
        }
    
    def _train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output, _ = self.model(data, return_components=False)
            
            # Compute loss with class weights
            loss = self._weighted_bce_loss(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predicted = (output > 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        f1 = self._calculate_f1(np.array(all_labels), np.array(all_preds))
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy,
            'f1': f1
        }
    
    def _validate_epoch(self, val_loader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output, _ = self.model(data, return_components=False)
                
                # Loss
                loss = self._weighted_bce_loss(output, target)
                total_loss += loss.item()
                
                # Metrics
                predicted = (output > 0.5).float()
                correct += (predicted == target).sum().item()
                total += target.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_scores.extend(output.cpu().numpy())
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        f1 = self._calculate_f1(np.array(all_labels), np.array(all_preds))
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'f1': f1
        }
    
    def _weighted_bce_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy with class weights."""
        # Calculate weights for each sample
        weights = torch.where(
            target == 1,
            torch.tensor(self.class_weight_positive, device=self.device),
            torch.tensor(1.0, device=self.device)
        )
        
        # BCE loss
        bce = nn.functional.binary_cross_entropy(output, target, reduction='none')
        
        # Apply weights
        weighted_bce = (bce * weights).mean()
        
        return weighted_bce
    
    def _prepare_sequences(
        self, 
        events: List[Dict[str, Any]], 
        sequence_length: int
    ) -> np.ndarray:
        """
        Convert events into sequences of features.
        
        Args:
            events: List of event dictionaries
            sequence_length: Length of each sequence
            
        Returns:
            Array of shape (num_sequences, sequence_length, num_features)
        """
        # Extract features for each event
        all_features = []
        for event in events:
            features = self.feature_extractor.extract_from_event(event)
            all_features.append(features)
        
        all_features = np.array(all_features)
        
        # Create sequences
        sequences = []
        for i in range(len(all_features) - sequence_length + 1):
            sequence = all_features[i:i + sequence_length]
            sequences.append(sequence)
        
        return np.array(sequences, dtype=np.float32)
    
    def _align_labels(self, labels: np.ndarray, num_sequences: int) -> np.ndarray:
        """Align labels with sequences (use label of last event in sequence)."""
        if len(labels) < num_sequences:
            # Pad with zeros if needed
            return np.pad(labels, (0, num_sequences - len(labels)), 'constant')
        else:
            # Use last labels
            return labels[-num_sequences:].reshape(-1, 1)
    
    def _create_dataloader(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool
    ) -> torch.utils.data.DataLoader:
        """Create PyTorch DataLoader."""
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(features),
            torch.FloatTensor(labels)
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 for compatibility
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def _calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score."""
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1
    
    def _save_checkpoint(
        self, 
        epoch: int, 
        metrics: Dict[str, float],
        is_best: bool = False,
        suffix: str = ""
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        filename = f"checkpoint_epoch_{epoch}{suffix}.pt"
        torch.save(checkpoint, self.checkpoint_dir / filename)
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pt")
            logger.info(f"Saved best model with val_loss: {metrics['loss']:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")
