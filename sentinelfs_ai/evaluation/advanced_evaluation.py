"""
Advanced evaluation metrics and validation approaches for model assessment.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score,
    brier_score_loss, precision_recall_fscore_support
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AdvancedEvaluator:
    """
    Advanced evaluator with comprehensive metrics and validation approaches.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_model_comprehensive(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation with advanced metrics.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            device: Evaluation device
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_data)
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Convert to numpy arrays
        y_pred_proba = np.array(all_outputs).flatten()
        y_true = np.array(all_labels).flatten()
        
        # Calculate comprehensive metrics only if we have multiple classes
        y_pred = (y_pred_proba >= threshold).astype(int)
        if len(np.unique(y_true)) < 2:
            # If only one class, return minimal metrics
            return {
                'accuracy': 1.0 if len(y_true) > 0 and np.all(y_true == y_pred) else 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'auc_roc': 0.5,  # Random classifier baseline
                'auc_pr': float(np.mean(y_true)) if len(y_true) > 0 else 0.0,  # Prior
                'mcc': 0.0,
                'kappa': 0.0,
                'calibration_error': 0.0,
                'log_loss': 0.0,  # This would be ideal if perfect
                'brier_score': 0.0,
                'balanced_accuracy': 0.5,  # Random baseline
                'confusion_matrix': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0},
                'true_positives': 0,
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0
            }
        
        metrics = self.calculate_comprehensive_metrics(y_true, y_pred_proba, threshold)
        return metrics
    
    def calculate_comprehensive_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics including advanced ones.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary with comprehensive metrics
        """
        # Convert probabilities to binary predictions (using provided threshold)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Basic metrics
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1:
            # Only one class present
            unique_val = y_true[0] if len(y_true) > 0 else 0
            if unique_val == 1:  # Only positives
                tp = cm[0, 0]
                tn = fp = fn = 0
            else:  # Only negatives
                tn = cm[0, 0]
                tp = fp = fn = 0
        else:
            # Handle other cases
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Advanced metrics
        try:
            auc_roc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
        except Exception as e:
            logger.warning(f"Could not calculate AUC-ROC: {e}")
            auc_roc = 0.5
        
        try:
            auc_pr = average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
        except Exception as e:
            logger.warning(f"Could not calculate AUC-PR: {e}")
            auc_pr = 0.5
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
        mcc = matthews_corrcoef(y_true, y_pred) if (tp + fp) > 0 and (tp + fn) > 0 and (tn + fp) > 0 and (tn + fn) > 0 else 0
        try:
            kappa = cohen_kappa_score(y_true, y_pred)
        except Exception:
            kappa = 0.0
        
        # Calibration error (lower is better)
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=10
            )
            calibration_err = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        except Exception:
            calibration_err = 0.0
        
        # Log loss (cross-entropy) - lower is better
        epsilon = 1e-15  # Small value to avoid log(0)
        y_pred_proba_clipped = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        log_loss = -np.mean(
            y_true * np.log(y_pred_proba_clipped) + 
            (1 - y_true) * np.log(1 - y_pred_proba_clipped)
        )
        
        # Brier score (for probabilistic predictions) - lower is better
        brier_score = brier_score_loss(y_true, y_pred_proba)
        
        # Balanced accuracy (accounting for class imbalance)
        balanced_acc = (recall + specificity) / 2
        
        # F1 at different thresholds
        f1_scores = []
        thresholds = [0.3, 0.4, threshold, 0.6, 0.7]
        for thresh in thresholds:
            y_pred_thresh = (y_pred_proba >= thresh).astype(int)
            tp_t = np.sum((y_true == 1) & (y_pred_thresh == 1))
            fp_t = np.sum((y_true == 0) & (y_pred_thresh == 1))
            fn_t = np.sum((y_true == 1) & (y_pred_thresh == 0))
            
            precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
            recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            f1_t = 2 * (precision_t * recall_t) / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0
            f1_scores.append(f1_t)
        
        # Precision at different recall levels (for PR curve interpretation)
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
            # Interpolate precision at recall levels 0.1, 0.3, 0.5, 0.7, 0.9
            recall_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
            precision_at_recall = []
            for level in recall_levels:
                idx = np.where(recall_vals >= level)[0]
                if len(idx) > 0:
                    precision_at_recall.append(float(np.max(precision_vals[idx])))
                else:
                    precision_at_recall.append(0.0)
        except Exception:
            precision_at_recall = [0.0] * 5
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'npv': float(npv),
            'auc_roc': float(auc_roc),
            'auc_pr': float(auc_pr),
            'mcc': float(mcc),
            'kappa': float(kappa),
            'calibration_error': float(calibration_err),
            'log_loss': float(log_loss),
            'brier_score': float(brier_score),
            'balanced_accuracy': float(balanced_acc),
            'f1_at_0.3': f1_scores[0],
            'f1_at_0.4': f1_scores[1], 
            'f1_at_threshold': f1_scores[2],  # The provided threshold
            'f1_at_0.6': f1_scores[3],
            'f1_at_0.7': f1_scores[4],
            'precision_at_0.1_recall': precision_at_recall[0],
            'precision_at_0.3_recall': precision_at_recall[1],
            'precision_at_0.5_recall': precision_at_recall[2],
            'precision_at_0.7_recall': precision_at_recall[3],
            'precision_at_0.9_recall': precision_at_recall[4],
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'threshold_used': threshold,
            'num_samples': int(len(y_true))
        }
    
    def cross_validate(
        self,
        model_class,
        model_params: Dict,
        data: np.ndarray,
        labels: np.ndarray,
        n_folds: int = 5,
        device: Optional[torch.device] = None,
        threshold: float = 0.5
    ) -> Dict[str, List[float]]:
        """
        Perform k-fold cross-validation.
        
        Args:
            model_class: Model class to instantiate
            model_params: Parameters for model initialization
            data: Input data
            labels: True labels
            n_folds: Number of folds
            device: Training device
            threshold: Classification threshold
            
        Returns:
            Dictionary with metrics for each fold
        """
        from sklearn.model_selection import StratifiedKFold
        from torch.utils.data import TensorDataset, DataLoader
        import torch.optim as optim
        
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert to tensors
        data_tensor = torch.FloatTensor(data)
        labels_tensor = torch.FloatTensor(labels)
        
        # Initialize k-fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Initialize results dictionary
        fold_results = {
            'fold': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'auc_roc': [],
            'auc_pr': [],
            'loss': [],
            'calibration_error': [],
            'brier_score': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels.flatten())):
            logger.info(f"Training fold {fold + 1}/{n_folds}")
            
            # Split data
            X_train = data_tensor[train_idx]
            y_train = labels_tensor[train_idx]
            X_val = data_tensor[val_idx]
            y_val = labels_tensor[val_idx]
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Instantiate and train model
            model = model_class(**model_params)
            model = model.to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.BCELoss()
            
            # Simple training loop for validation (you may want to adjust this based on your needs)
            model.train()
            for epoch in range(10):  # Limited epochs for validation
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    targets = batch_y.unsqueeze(1)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                val_outputs = []
                val_labels = []
                
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    outputs = model(batch_x)
                    val_outputs.extend(outputs.cpu().numpy())
                    val_labels.extend(batch_y.cpu().numpy())
                
                val_outputs = np.array(val_outputs).flatten()
                val_labels = np.array(val_labels).flatten()
                
                # Calculate metrics
                fold_metrics = self.calculate_comprehensive_metrics(val_labels, val_outputs, threshold)
                
                fold_results['fold'].append(fold)
                fold_results['accuracy'].append(fold_metrics['accuracy'])
                fold_results['precision'].append(fold_metrics['precision'])
                fold_results['recall'].append(fold_metrics['recall'])
                fold_results['f1_score'].append(fold_metrics['f1_score'])
                fold_results['auc_roc'].append(fold_metrics['auc_roc'])
                fold_results['auc_pr'].append(fold_metrics['auc_pr'])
                fold_results['loss'].append(fold_metrics['log_loss'])
                fold_results['calibration_error'].append(fold_metrics['calibration_error'])
                fold_results['brier_score'].append(fold_metrics['brier_score'])
        
        return fold_results
    
    def stratified_evaluation(
        self,
        model: torch.nn.Module,
        data: np.ndarray,
        labels: np.ndarray,
        anomaly_types: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
        threshold: float = 0.5
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform stratified evaluation by anomaly type or other categories.
        
        Args:
            model: Model to evaluate
            data: Input data
            labels: True labels
            anomaly_types: Optional anomaly type labels for stratification
            device: Evaluation device
            threshold: Classification threshold
            
        Returns:
            Dictionary with metrics for each stratum
        """
        from torch.utils.data import TensorDataset, DataLoader
        
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        # If no anomaly types provided, just evaluate overall and by label
        if anomaly_types is None:
            # Overall evaluation
            dataset = TensorDataset(torch.FloatTensor(data), torch.FloatTensor(labels))
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            results = {
                'overall': self.evaluate_model_comprehensive(model, loader, device, threshold)
            }
            
            # Evaluate only on anomalies
            anomaly_mask = (labels.flatten() == 1)
            if np.any(anomaly_mask):
                anomaly_data = data[anomaly_mask]
                anomaly_labels = labels[anomaly_mask]
                
                anomaly_dataset = TensorDataset(
                    torch.FloatTensor(anomaly_data), 
                    torch.FloatTensor(anomaly_labels)
                )
                anomaly_loader = DataLoader(anomaly_dataset, batch_size=32, shuffle=False)
                
                results['anomaly_only'] = self.evaluate_model_comprehensive(model, anomaly_loader, device, threshold)
            
            # Evaluate only on normal samples
            normal_mask = (labels.flatten() == 0)
            if np.any(normal_mask):
                normal_data = data[normal_mask]
                normal_labels = labels[normal_mask]
                
                normal_dataset = TensorDataset(
                    torch.FloatTensor(normal_data), 
                    torch.FloatTensor(normal_labels)
                )
                normal_loader = DataLoader(normal_dataset, batch_size=32, shuffle=False)
                
                results['normal_only'] = self.evaluate_model_comprehensive(model, normal_loader, device, threshold)
                
        else:
            # Evaluate by anomaly type
            results = {}
            
            unique_types = np.unique(anomaly_types)
            for anomaly_type in unique_types:
                type_mask = (anomaly_types.flatten() == anomaly_type)
                type_data = data[type_mask]
                type_labels = labels[type_mask]
                
                if len(type_data) == 0:
                    continue
                
                type_dataset = TensorDataset(
                    torch.FloatTensor(type_data), 
                    torch.FloatTensor(type_labels)
                )
                type_loader = DataLoader(type_dataset, batch_size=32, shuffle=False)
                
                results[f'anomaly_type_{anomaly_type}'] = self.evaluate_model_comprehensive(
                    model, type_loader, device, threshold
                )
        
        return results
    
    def temporal_validation(
        self,
        model: torch.nn.Module,
        data: np.ndarray,
        labels: np.ndarray,
        time_stamps: np.ndarray,
        device: Optional[torch.device] = None,
        threshold: float = 0.5
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform temporal validation to check model performance over time.
        
        Args:
            model: Model to evaluate
            data: Input data
            labels: True labels
            time_stamps: Time stamps for temporal ordering
            device: Evaluation device
            threshold: Classification threshold
            
        Returns:
            Dictionary with metrics for different time periods
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Sort data by time stamps
        sorted_indices = np.argsort(time_stamps)
        sorted_data = data[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        # Split into time periods (e.g., early, middle, late)
        n_samples = len(sorted_data)
        period_size = n_samples // 3
        
        results = {}
        
        for period, (start, end) in enumerate([
            (0, period_size),
            (period_size, 2 * period_size),
            (2 * period_size, n_samples)
        ]):
            period_data = sorted_data[start:end]
            period_labels = sorted_labels[start:end]
            
            if len(period_data) == 0:
                continue
            
            # Create dataset and loader for this period
            dataset = TensorDataset(
                torch.FloatTensor(period_data), 
                torch.FloatTensor(period_labels)
            )
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            results[f'period_{period}'] = self.evaluate_model_comprehensive(model, loader, device, threshold)
        
        return results
    
    def get_performance_by_confidence(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None,
        n_bins: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance across different confidence levels.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            device: Evaluation device
            n_bins: Number of confidence bins to evaluate
            
        Returns:
            Dictionary with performance metrics for each confidence bin
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_data)
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        y_pred_proba = np.array(all_outputs).flatten()
        y_true = np.array(all_labels).flatten()
        
        # Create confidence bins
        results = {}
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            lower = bin_edges[i]
            upper = bin_edges[i+1]
            
            # Select predictions within this confidence range
            if i == n_bins - 1:  # Last bin includes upper bound
                mask = (y_pred_proba >= lower) & (y_pred_proba <= upper)
            else:  # Other bins exclude upper bound
                mask = (y_pred_proba >= lower) & (y_pred_proba < upper)
            
            if np.sum(mask) == 0:
                continue
                
            y_true_subset = y_true[mask]
            y_pred_proba_subset = y_pred_proba[mask]
            
            if len(y_true_subset) > 0:
                subset_metrics = self.calculate_comprehensive_metrics(
                    y_true_subset, 
                    y_pred_proba_subset
                )
                results[f'confidence_{lower:.1f}-{upper:.1f}'] = subset_metrics
        
        return results
    
    def detect_overfitting(
        self,
        train_metrics: Dict[str, List[float]],
        val_metrics: Dict[str, List[float]]
    ) -> Dict[str, bool]:
        """
        Detect signs of overfitting by comparing training and validation metrics.
        
        Args:
            train_metrics: Training metrics history (dictionary with lists)
            val_metrics: Validation metrics history (dictionary with lists)
            
        Returns:
            Dictionary with overfitting detection results
        """
        results = {}
        
        # Check if training accuracy is significantly higher than validation accuracy
        if 'train_acc' in train_metrics and 'val_acc' in val_metrics:
            train_acc_final = train_metrics['train_acc'][-1] if train_metrics['train_acc'] else 0
            val_acc_final = val_metrics['val_acc'][-1] if val_metrics['val_acc'] else 0
            results['accuracy_overfitting'] = train_acc_final - val_acc_final > 0.1
            
            # Also check if validation accuracy is decreasing
            if len(val_metrics['val_acc']) >= 3:
                recent_val_acc = val_metrics['val_acc'][-3:]
                results['decreasing_val_accuracy'] = recent_val_acc[-1] < recent_val_acc[0]
        
        # Check if training loss is significantly lower than validation loss
        if 'train_loss' in train_metrics and 'val_loss' in val_metrics:
            train_loss_final = train_metrics['train_loss'][-1] if train_metrics['train_loss'] else 1
            val_loss_final = val_metrics['val_loss'][-1] if val_metrics['val_loss'] else 1
            # If training loss is much lower than validation loss, it may indicate overfitting
            results['loss_overfitting'] = val_loss_final / (train_loss_final + 1e-8) > 2.0
        
        # Check for other overfitting indicators
        results['any_overfitting'] = any(results.values())
        
        return results


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, title: str = "ROC Curve", save_path: Optional[str] = None):
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, title: str = "Precision-Recall Curve", save_path: Optional[str] = None):
    """Plot Precision-Recall curve."""
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='b', lw=2, 
             label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix", save_path: Optional[str] = None, normalize: bool = False):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title += " (Normalized)"
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'], 
                yticklabels=['Normal', 'Anomaly'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def calibration_plot(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10, save_path: Optional[str] = None):
    """Plot calibration curve."""
    from sklearn.calibration import calibration_curve
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, title: str = "ROC Curve"):
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, title: str = "Precision-Recall Curve"):
    """Plot Precision-Recall curve."""
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='b', lw=2, 
             label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'], 
                yticklabels=['Normal', 'Anomaly'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def calibration_plot(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10):
    """Plot calibration curve."""
    from sklearn.calibration import calibration_curve
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend()
    plt.grid(True)
    plt.show()