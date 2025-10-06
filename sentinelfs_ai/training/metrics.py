"""
Performance metrics calculation for model evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict


def calculate_metrics(
    predictions: np.ndarray, 
    labels: np.ndarray, 
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        predictions: Model predictions (probabilities)
        labels: True labels
        threshold: Classification threshold
        
    Returns:
        Dictionary of metric values
    """
    pred_binary = (predictions >= threshold).astype(int)
    
    # Calculate confusion matrix components
    tp = np.sum((pred_binary == 1) & (labels == 1))
    tn = np.sum((pred_binary == 0) & (labels == 0))
    fp = np.sum((pred_binary == 1) & (labels == 0))
    fn = np.sum((pred_binary == 0) & (labels == 1))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def evaluate_model(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> tuple:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        
    Returns:
        Tuple of (loss, metrics)
    """
    model.eval()
    criterion = nn.BCELoss()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_preds), np.array(all_labels))
    
    return avg_loss, metrics
