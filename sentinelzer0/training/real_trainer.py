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
from copy import deepcopy

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score
)

from ..models.hybrid_detector import HybridThreatDetector
from ..data.real_feature_extractor import RealFeatureExtractor
from ..utils.logger import get_logger
from ..utils.device import get_device
from ..mlops.mlflow_integration import MLflowTracker

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
    - Adaptive decision threshold optimization
    - Comprehensive metric tracking (precision, recall, ROC-AUC, PR-AUC)
    - Dynamic class rebalancing with weighted sampling
    - Optional MLflow experiment tracking
    """
    
    def __init__(
        self,
        model: HybridThreatDetector,
        feature_extractor: RealFeatureExtractor,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        class_weight_positive: float = 2.0,  # Weight for positive (anomaly) class
        patience: int = 10,
        checkpoint_dir: str = './checkpoints',
        decision_threshold: float = 0.5,
        early_stopping_metric: str = 'f1',
        min_delta: float = 1e-4,
        balance_classes: bool = True,
        min_positive_fraction: float = 0.05,
        dynamic_class_weighting: bool = True,
        enable_mlflow: bool = False,
        mlflow_tracker: Optional[MLflowTracker] = None,
        mlflow_experiment: str = "SentinelZer0"
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = get_device()
        self.model = self.model.to(self.device)
        self.initial_lr = learning_rate
        self.weight_decay = weight_decay
        self.balance_classes = balance_classes
        self.dynamic_class_weighting = dynamic_class_weighting
        self.min_positive_fraction = min_positive_fraction
        self.base_class_weight_positive = float(class_weight_positive)
        self.positive_weight_floor = max(1.0, self.base_class_weight_positive)
        
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
        self.class_weight_positive = float(class_weight_positive)
        
        # Early stopping
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        self.early_stopping_metric = early_stopping_metric.lower()
        if self.early_stopping_metric not in {'loss', 'f1', 'precision', 'recall', 'auc', 'pr_auc'}:
            raise ValueError(
                "early_stopping_metric must be one of: 'loss', 'f1', 'precision', 'recall', 'auc', 'pr_auc'"
            )
        self.monitor_mode = 'min' if self.early_stopping_metric == 'loss' else 'max'
        self.min_delta = min_delta
        self.best_score = float('inf') if self.monitor_mode == 'min' else float('-inf')
        self.best_epoch = None
        self.decision_threshold = float(decision_threshold)
        self.best_threshold = self.decision_threshold
        
        # MLflow integration
        self.mlflow_tracker = mlflow_tracker
        self.mlflow_experiment = mlflow_experiment
        self.enable_mlflow = False
        self._mlflow_run_active = False
        self._mlflow_logged_params = False
        self.mlflow_run_name = None
        self.mlflow_run_tags = {'trainer': 'RealWorldTrainer'}
        if enable_mlflow and self.mlflow_tracker is None:
            try:
                self.mlflow_tracker = MLflowTracker(experiment_name=mlflow_experiment)
            except Exception as exc:
                logger.warning(f"Unable to initialize MLflow tracker: {exc}")
                self.mlflow_tracker = None
        if self.mlflow_tracker is not None and getattr(self.mlflow_tracker, 'is_available', lambda: False)():
            self.enable_mlflow = True
        elif enable_mlflow:
            logger.warning("MLflow logging requested but tracker unavailable. Proceeding without MLflow.")
        
        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'train_f1': [],
            'val_f1': [],
            'train_auc': [],
            'val_auc': [],
            'train_pr_auc': [],
            'val_pr_auc': [],
            'val_threshold': [],
            'learning_rates': []
        }
        
        # Performance metrics
        self.metrics = {
            'best_epoch': None,
            'best_threshold': self.decision_threshold,
            'best_val_metrics': {},
            'best_score': self.best_score,
            'train_class_balance': {},
            'val_class_balance': {},
            'sampler': {'applied': False},
            'mlflow_enabled': self.enable_mlflow,
            'class_weight_positive': self.class_weight_positive,
            'incremental': {}
        }
    
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
        train_balance = self._update_class_balance_stats(
            train_labels_aligned,
            split='train',
            update_weight=True
        )
        val_balance = self._update_class_balance_stats(
            val_labels_aligned,
            split='val',
            update_weight=False
        )
        logger.info(
            "Class balance - train: pos_fraction=%.4f, val: pos_fraction=%.4f, positive_weight=%.3f",
            train_balance['positive_fraction'],
            val_balance['positive_fraction'],
            self.class_weight_positive
        )
        
        # Create data loaders
        train_loader = self._create_dataloader(
            train_features, train_labels_aligned, batch_size, shuffle=True
        )
        val_loader = self._create_dataloader(
            val_features, val_labels_aligned, batch_size, shuffle=False
        )
        
        # Training loop
        start_time = time.time()
        self._start_mlflow_run()
        
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
            self.history['train_precision'].append(train_metrics['precision'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['train_recall'].append(train_metrics['recall'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['train_auc'].append(train_metrics['roc_auc'])
            self.history['val_auc'].append(val_metrics['roc_auc'])
            self.history['train_pr_auc'].append(train_metrics['pr_auc'])
            self.history['val_pr_auc'].append(val_metrics['pr_auc'])
            self.history['val_threshold'].append(val_metrics.get('optimal_threshold', self.decision_threshold))
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Early stopping check
            monitor_value = self._get_monitor_value(val_metrics)
            if self._is_new_best(monitor_value):
                self.best_score = monitor_value
                self.patience_counter = 0
                self.best_model_state = deepcopy(self.model.state_dict())
                self.best_val_loss = min(self.best_val_loss, val_metrics['loss'])
                self.best_threshold = val_metrics.get('optimal_threshold', self.best_threshold)
                self.metrics.update({
                    'best_epoch': epoch + 1,
                    'best_threshold': self.best_threshold,
                    'best_val_metrics': deepcopy(val_metrics.get('best_metrics', val_metrics)),
                    'best_score': self.best_score
                })
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
            
            # Log progress
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s) - "
                f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, Val Precision: {val_metrics['precision']:.4f}, "
                f"Val Recall: {val_metrics['recall']:.4f}, Val F1: {val_metrics['f1']:.4f}, "
                f"Val AUC: {val_metrics['roc_auc']:.4f}, Opt Threshold: {val_metrics.get('optimal_threshold', self.decision_threshold):.3f}"
            )
            self._log_epoch_metrics(epoch, train_metrics, val_metrics, epoch_time)
            
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
        self.metrics['training_time'] = training_time
        
        # Save final model components
        self.model.save_components(str(self.checkpoint_dir / 'final'))
        
        # Finalize metrics
        self.decision_threshold = self.best_threshold
        self.history['best_val_f1'] = self.metrics['best_val_metrics'].get('f1', 0.0)
        self.history['best_epoch'] = self.metrics.get('best_epoch', None)
        self.history['best_threshold'] = self.decision_threshold
        self.history['best_score'] = self.metrics.get('best_score', None)
        self._close_mlflow_run()
        
        return {
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time,
            'final_metrics': val_metrics,
            'best_metrics': self.metrics,
            'decision_threshold': self.decision_threshold
        }
    
    def incremental_update(
        self,
        new_events: List[Dict[str, Any]],
        new_labels: np.ndarray,
        num_epochs: int = 5,
        batch_size: int = 32,
        sequence_length: int = 50
    ) -> Dict[str, Any]:
        """
        Perform incremental learning on new data.
        
        Args:
            new_events: New file system events
            new_labels: Labels for new events
            num_epochs: Number of update epochs
            batch_size: Batch size
            sequence_length: Sequence length
            
        Returns:
            Dictionary containing training averages, evaluation metrics, class balance, and threshold update info
        """
        logger.info(f"Performing incremental update with {len(new_events)} new events...")
        
        # Extract features
        new_features = self._prepare_sequences(new_events, sequence_length)
        new_labels_aligned = self._align_labels(new_labels, len(new_features))
        balance_stats = self._update_class_balance_stats(
            new_labels_aligned,
            split='incremental',
            update_weight=True
        )
        logger.info(
            "Incremental class balance - pos_fraction=%.4f, positive_weight=%.3f",
            balance_stats['positive_fraction'],
            self.class_weight_positive
        )
        if len(new_features) == 0:
            logger.warning("No sequences generated from incremental data; skipping update.")
            return {
                'num_samples': 0,
                'training_metrics': {},
                'evaluation_metrics': {},
                'class_balance': balance_stats,
                'threshold_updated': False,
                'new_threshold': self.decision_threshold
            }
        
        # Create data loaders (training with sampling, evaluation without)
        update_loader = self._create_dataloader(
            new_features, new_labels_aligned, batch_size, shuffle=True
        )
        eval_loader = self._create_dataloader(
            new_features, new_labels_aligned, batch_size, shuffle=False
        )
        
        # Fine-tune model with lower learning rate
        old_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer.param_groups[0]['lr'] = old_lr * 0.1  # 10x smaller LR
        
        # Update for a few epochs
        total_loss = 0.0
        total_acc = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_auc = 0.0
        total_pr_auc = 0.0
        effective_epochs = max(num_epochs, 1)
        
        for epoch in range(effective_epochs):
            metrics = self._train_epoch(update_loader, epoch)
            total_loss += metrics['loss']
            total_acc += metrics['accuracy']
            total_precision += metrics['precision']
            total_recall += metrics['recall']
            total_f1 += metrics['f1']
            total_auc += metrics['roc_auc']
            total_pr_auc += metrics['pr_auc']
        
        # Restore learning rate
        self.optimizer.param_groups[0]['lr'] = old_lr
        
        # Evaluate updated model on incremental data
        eval_metrics = self._validate_epoch(eval_loader, epoch=-1)
        logger.info(
            "Incremental evaluation - F1: %.4f (optimal %.4f), Precision: %.4f, Recall: %.4f, Threshold: %.3f",
            eval_metrics['f1'],
            eval_metrics.get('optimal_f1', eval_metrics['f1']),
            eval_metrics['precision'],
            eval_metrics['recall'],
            eval_metrics.get('optimal_threshold', self.decision_threshold)
        )
        
        avg_metrics = {
            'avg_loss': total_loss / effective_epochs,
            'avg_accuracy': total_acc / effective_epochs,
            'avg_precision': total_precision / effective_epochs,
            'avg_recall': total_recall / effective_epochs,
            'avg_f1': total_f1 / effective_epochs,
            'avg_auc': total_auc / effective_epochs,
            'avg_pr_auc': total_pr_auc / effective_epochs
        }
        incremental_summary = {
            'training': avg_metrics,
            'evaluation': eval_metrics,
            'num_samples': len(new_features),
            'class_balance': balance_stats
        }
        self.metrics['incremental'] = incremental_summary
        
        current_best_f1 = self.metrics.get('best_val_metrics', {}).get('f1', 0.0)
        threshold_updated = False
        if eval_metrics.get('optimal_f1', 0.0) > current_best_f1 + 1e-6:
            self.best_threshold = eval_metrics.get('optimal_threshold', self.best_threshold)
            self.decision_threshold = self.best_threshold
            self.metrics['best_threshold'] = self.best_threshold
            self.metrics['best_val_metrics'] = deepcopy(eval_metrics.get('best_metrics', eval_metrics))
            self.metrics['best_score'] = eval_metrics.get('optimal_f1', current_best_f1)
            threshold_updated = True
            logger.info(
                "Updated decision threshold to %.3f based on incremental evaluation (F1=%.4f)",
                self.best_threshold,
                self.metrics['best_score']
            )
        
        # Save updated model snapshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_checkpoint(-1, eval_metrics, is_best=False, suffix=f"_incremental_{timestamp}")
        
        return {
            'num_samples': len(new_features),
            'training_metrics': avg_metrics,
            'evaluation_metrics': eval_metrics,
            'class_balance': balance_stats,
            'threshold_updated': threshold_updated,
            'new_threshold': self.decision_threshold
        }
    
    def _train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_scores: List[float] = []
        all_labels: List[float] = []
        
        for data, target in train_loader:
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
            all_scores.extend(output.detach().cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy().flatten())
        
        metrics = self._calculate_classification_metrics(
            np.array(all_labels),
            np.array(all_scores),
            threshold=self.decision_threshold
        )
        metrics['loss'] = total_loss / max(len(train_loader), 1)
        
        return metrics
    
    def _validate_epoch(self, val_loader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        all_scores: List[float] = []
        all_labels: List[float] = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output, _ = self.model(data, return_components=False)
                
                # Loss
                loss = self._weighted_bce_loss(output, target)
                total_loss += loss.item()
                
                all_scores.extend(output.detach().cpu().numpy().flatten())
                all_labels.extend(target.cpu().numpy().flatten())
        
        labels_array = np.array(all_labels)
        scores_array = np.array(all_scores)
        metrics = self._calculate_classification_metrics(
            labels_array,
            scores_array,
            threshold=self.decision_threshold
        )
        metrics['loss'] = total_loss / max(len(val_loader), 1)
        optimal_threshold, optimal_metrics = self._find_optimal_threshold(labels_array, scores_array)
        metrics['optimal_threshold'] = optimal_threshold
        metrics['optimal_f1'] = optimal_metrics.get('f1', metrics.get('f1', 0.0))
        metrics['optimal_precision'] = optimal_metrics.get('precision', metrics.get('precision', 0.0))
        metrics['optimal_recall'] = optimal_metrics.get('recall', metrics.get('recall', 0.0))
        metrics['optimal_auc'] = optimal_metrics.get('roc_auc', metrics.get('roc_auc', 0.0))
        metrics['optimal_pr_auc'] = optimal_metrics.get('pr_auc', metrics.get('pr_auc', 0.0))
        optimal_metrics_with_loss = optimal_metrics.copy()
        optimal_metrics_with_loss['loss'] = metrics['loss']
        metrics['best_metrics'] = optimal_metrics_with_loss
        
        return metrics
    
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

    def _calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """Compute classification metrics for given labels and scores."""
        if threshold is None:
            threshold = self.decision_threshold
        
        if y_true.size == 0 or scores.size == 0:
            return {
                'loss': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'roc_auc': 0.5,
                'pr_auc': 0.0,
                'support_pos': 0,
                'support_neg': 0
            }
        
        y_true = y_true.flatten().astype(int)
        scores = np.clip(scores.flatten(), 0.0, 1.0)
        preds = (scores >= threshold).astype(int)
        accuracy = float((preds == y_true).mean()) if y_true.size > 0 else 0.0
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            preds,
            average='binary',
            zero_division=0
        )
        unique_labels = np.unique(y_true)
        if unique_labels.size < 2:
            roc_auc = 0.5
        else:
            roc_auc = self._safe_metric(roc_auc_score, y_true, scores, default=0.5)
        pr_auc = self._safe_metric(average_precision_score, y_true, scores, default=0.0)
        
        return {
            'accuracy': accuracy,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'support_pos': int((y_true == 1).sum()),
            'support_neg': int((y_true == 0).sum())
        }

    def _safe_metric(self, metric_fn, y_true: np.ndarray, scores: np.ndarray, default: float = 0.0) -> float:
        """Safely compute a metric, returning default when invalid."""
        try:
            value = metric_fn(y_true, scores)
            if isinstance(value, np.ndarray):
                value = value.item()
            return float(value)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return float(default)

    def _find_optimal_threshold(
        self,
        y_true: np.ndarray,
        scores: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Determine threshold that maximizes F1, breaking ties by recall."""
        if y_true.size == 0 or scores.size == 0:
            default_metrics = self._calculate_classification_metrics(y_true, scores, threshold=self.decision_threshold)
            return self.decision_threshold, default_metrics
        
        thresholds = self._generate_threshold_candidates(scores)
        best_threshold = self.decision_threshold
        best_metrics = self._calculate_classification_metrics(y_true, scores, threshold=best_threshold)
        
        for threshold in thresholds:
            metrics = self._calculate_classification_metrics(y_true, scores, threshold=float(threshold))
            if metrics['f1'] > best_metrics['f1'] + 1e-6:
                best_threshold = float(threshold)
                best_metrics = metrics
            elif abs(metrics['f1'] - best_metrics['f1']) <= 1e-6 and metrics['recall'] > best_metrics['recall'] + 1e-6:
                best_threshold = float(threshold)
                best_metrics = metrics
        
        best_metrics = best_metrics.copy()
        best_metrics['threshold'] = best_threshold
        return best_threshold, best_metrics

    def _generate_threshold_candidates(self, scores: np.ndarray) -> np.ndarray:
        """Generate diverse threshold candidates between observed score bounds."""
        scores = np.clip(scores.flatten(), 0.0, 1.0)
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        if min_score == max_score:
            return np.array([self.decision_threshold], dtype=np.float32)
        grid = np.linspace(0.05, 0.95, num=19)
        adaptive = np.linspace(min_score, max_score, num=25)
        thresholds = np.unique(np.concatenate([grid, adaptive, [self.decision_threshold]]))
        thresholds = thresholds[(thresholds >= 0.0) & (thresholds <= 1.0)]
        return thresholds

    def _get_monitor_value(self, val_metrics: Dict[str, float]) -> float:
        """Extract monitored metric for early stopping."""
        metric = self.early_stopping_metric
        if metric == 'loss':
            return float(val_metrics.get('loss', float('inf')))
        if metric == 'f1':
            return float(val_metrics.get('optimal_f1', val_metrics.get('f1', 0.0)))
        if metric == 'precision':
            return float(val_metrics.get('optimal_precision', val_metrics.get('precision', 0.0)))
        if metric == 'recall':
            return float(val_metrics.get('optimal_recall', val_metrics.get('recall', 0.0)))
        if metric == 'auc':
            return float(val_metrics.get('optimal_auc', val_metrics.get('roc_auc', 0.0)))
        if metric == 'pr_auc':
            return float(val_metrics.get('optimal_pr_auc', val_metrics.get('pr_auc', 0.0)))
        return float(val_metrics.get('f1', 0.0))

    def _is_new_best(self, value: float) -> bool:
        """Check if new value improves upon best score."""
        if value is None or np.isnan(value):
            return False
        if self.monitor_mode == 'min':
            return value < (self.best_score - self.min_delta)
        return value > (self.best_score + self.min_delta)

    def _update_class_balance_stats(
        self,
        labels: np.ndarray,
        split: str = 'train',
        update_weight: bool = False
    ) -> Dict[str, Any]:
        """Record class balance statistics and update positive class weight if needed."""
        labels_flat = labels.flatten().astype(int)
        total = int(labels_flat.size)
        pos = int(np.sum(labels_flat == 1))
        neg = int(np.sum(labels_flat == 0))
        positive_fraction = (pos / total) if total > 0 else 0.0
        stats = {
            'positive_fraction': positive_fraction,
            'positive_count': pos,
            'negative_count': neg,
            'total_samples': total,
            'class_weight_positive': self.class_weight_positive
        }
        if update_weight and self.dynamic_class_weighting:
            previous_weight = self.class_weight_positive
            new_weight = self._compute_dynamic_positive_weight(pos, neg)
            if abs(new_weight - previous_weight) > 1e-6:
                logger.info(
                    "Adjusted positive class weight from %.3f to %.3f (train positive fraction=%.4f)",
                    previous_weight,
                    new_weight,
                    positive_fraction
                )
            self.class_weight_positive = new_weight
            stats['class_weight_positive'] = new_weight
        self.metrics[f'{split}_class_balance'] = stats
        self.metrics['class_weight_positive'] = self.class_weight_positive
        return stats

    def _compute_dynamic_positive_weight(self, positive_count: int, negative_count: int) -> float:
        """Compute a stable positive class weight based on class distribution."""
        if positive_count <= 0:
            return max(self.positive_weight_floor, self.base_class_weight_positive)
        ratio = negative_count / positive_count if positive_count > 0 else self.positive_weight_floor
        ratio = max(self.positive_weight_floor, ratio)
        ratio = min(ratio, 50.0)  # Prevent extreme scaling
        return float(ratio)

    def _build_sample_weights(
        self,
        labels: np.ndarray
    ) -> Tuple[Optional[torch.DoubleTensor], Dict[str, Any]]:
        """Create sample weights for balanced sampling when data is imbalanced."""
        labels_flat = labels.flatten().astype(int)
        total = int(labels_flat.size)
        metadata: Dict[str, Any] = {
            'applied': False,
            'positive_fraction': 0.0,
            'positive_count': 0,
            'negative_count': 0,
            'total_samples': total,
            'min_positive_fraction': self.min_positive_fraction,
            'weight_positive': self.class_weight_positive,
            'weight_negative': 1.0
        }
        if total == 0:
            return None, metadata
        pos = int(np.sum(labels_flat == 1))
        neg = int(np.sum(labels_flat == 0))
        positive_fraction = (pos / total) if total > 0 else 0.0
        metadata.update({
            'positive_fraction': positive_fraction,
            'positive_count': pos,
            'negative_count': neg
        })
        if pos == 0 or neg == 0:
            return None, metadata
        imbalance = (
            positive_fraction < self.min_positive_fraction or
            positive_fraction > (1.0 - self.min_positive_fraction)
        )
        if not imbalance:
            return None, metadata
        weight_negative = total / (2.0 * max(neg, 1))
        weight_positive = total / (2.0 * max(pos, 1))
        weight_positive = max(weight_positive, self.class_weight_positive, self.positive_weight_floor)
        sample_weights = np.where(labels_flat == 1, weight_positive, weight_negative).astype(np.float64)
        metadata.update({
            'applied': True,
            'weight_positive': float(weight_positive),
            'weight_negative': float(weight_negative)
        })
        return torch.DoubleTensor(sample_weights), metadata

    def _start_mlflow_run(self):
        """Start an MLflow run when tracking is enabled."""
        if not self.enable_mlflow or self.mlflow_tracker is None or self._mlflow_run_active:
            return
        run_name = f"real_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            self.mlflow_tracker.start_run(run_name=run_name, tags=self.mlflow_run_tags)
            self.mlflow_run_name = run_name
            self._mlflow_run_active = True
            if not self._mlflow_logged_params:
                params = {
                    'learning_rate': self.initial_lr,
                    'weight_decay': self.weight_decay,
                    'class_weight_positive_start': self.base_class_weight_positive,
                    'dynamic_class_weighting': self.dynamic_class_weighting,
                    'balance_classes': self.balance_classes,
                    'patience': self.patience,
                    'early_stopping_metric': self.early_stopping_metric,
                    'min_delta': self.min_delta
                }
                self.mlflow_tracker.log_params(params)
                self._mlflow_logged_params = True
        except Exception as exc:
            logger.warning(f"Failed to start MLflow run: {exc}")
            self.enable_mlflow = False

    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float
    ):
        """Log epoch metrics to MLflow if enabled."""
        if not self.enable_mlflow or self.mlflow_tracker is None or not self._mlflow_run_active:
            return
        payload: Dict[str, float] = {}
        ignore_keys = {'support_pos', 'support_neg'}
        for prefix, metrics in (('train', train_metrics), ('val', val_metrics)):
            for key, value in metrics.items():
                if key in ignore_keys or not isinstance(value, (int, float)):
                    continue
                if np.isnan(value):
                    continue
                payload[f"{prefix}/{key}"] = float(value)
        payload['epoch_time'] = float(epoch_time)
        payload['learning_rate'] = float(self.optimizer.param_groups[0]['lr'])
        try:
            self.mlflow_tracker.log_metrics(payload, step=epoch + 1)
        except Exception as exc:
            logger.warning(f"Failed to log MLflow metrics: {exc}")
            self.enable_mlflow = False

    def _close_mlflow_run(self):
        """Finalize MLflow run with summary metrics."""
        if not self.enable_mlflow or self.mlflow_tracker is None or not self._mlflow_run_active:
            return
        summary = {}
        best_metrics = self.metrics.get('best_val_metrics', {})
        summary['summary/best_val_f1'] = float(best_metrics.get('f1', 0.0))
        summary['summary/best_val_precision'] = float(best_metrics.get('precision', 0.0))
        summary['summary/best_val_recall'] = float(best_metrics.get('recall', 0.0))
        summary['summary/best_val_auc'] = float(best_metrics.get('roc_auc', 0.0))
        summary['summary/best_val_pr_auc'] = float(best_metrics.get('pr_auc', 0.0))
        summary['summary/best_threshold'] = float(self.metrics.get('best_threshold', self.decision_threshold))
        summary['summary/best_epoch'] = float(self.metrics.get('best_epoch') or 0)
        summary['summary/training_time'] = float(self.metrics.get('training_time', 0.0))
        try:
            self.mlflow_tracker.log_metrics(summary)
        except Exception as exc:
            logger.warning(f"Failed to log MLflow summary metrics: {exc}")
        finally:
            try:
                self.mlflow_tracker.end_run()
            except Exception as exc:
                logger.warning(f"Failed to close MLflow run: {exc}")
            self._mlflow_run_active = False
    
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
        
        sampler = None
        if self.balance_classes and shuffle:
            sample_weights, sampler_info = self._build_sample_weights(labels)
            self.metrics['sampler'] = sampler_info
            if sample_weights is not None and sampler_info.get('applied', False):
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                shuffle = False
                logger.info(
                    "Using weighted sampler (pos_fraction=%.4f, weight_pos=%.2f, weight_neg=%.2f)",
                    sampler_info.get('positive_fraction', 0.0),
                    sampler_info.get('weight_positive', self.class_weight_positive),
                    sampler_info.get('weight_negative', 1.0)
                )
        else:
            self.metrics['sampler'] = {'applied': False}
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=0,  # Set to 0 for compatibility
            pin_memory=True if torch.cuda.is_available() else False
        )
    
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
