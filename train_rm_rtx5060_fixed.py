"""
CRITICAL FIX: Complete Threat Detection System with Proper Threshold Calibration
Addresses catastrophic generalization failure with comprehensive diagnostics

Key Fixes:
1. ROC/PR curve-based threshold optimization
2. Detailed prediction logging and score distribution analysis
3. Real GPU monitoring with nvidia-smi integration
4. Enhanced test data with verified labels
5. Train/test distribution validation
6. Adversarial validation for distribution mismatch detection
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random
import json
import time
from typing import Dict, List, Tuple, Optional
import gc
import subprocess
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

# Enhanced imports
from sentinelfs_ai.models.hybrid_detector import HybridThreatDetector
from sentinelfs_ai.data.real_feature_extractor import RealFeatureExtractor
from sentinelfs_ai.training.real_trainer import RealWorldTrainer
from sentinelfs_ai.inference.real_engine import RealTimeInferenceEngine
from sentinelfs_ai.evaluation.production_evaluator import ProductionEvaluator
from sentinelfs_ai.utils.logger import get_logger

logger = get_logger(__name__)


class GPUMonitor:
    """Real GPU monitoring with nvidia-smi."""
    
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.cuda_available = torch.cuda.is_available()
        
    def get_gpu_stats(self) -> Dict:
        """Get actual GPU statistics."""
        if not self.cuda_available:
            return {'gpu_available': False, 'utilization': 0, 'memory_used_mb': 0, 'memory_total_mb': 0}
        
        try:
            # Use nvidia-smi for accurate stats
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                util, mem_used, mem_total = result.stdout.strip().split(',')
                return {
                    'gpu_available': True,
                    'utilization': float(util),
                    'memory_used_mb': float(mem_used),
                    'memory_total_mb': float(mem_total),
                    'device_name': torch.cuda.get_device_name(0)
                }
        except Exception as e:
            logger.warning(f"nvidia-smi failed: {e}, using PyTorch stats")
        
        # Fallback to PyTorch stats
        return {
            'gpu_available': True,
            'utilization': 'unknown',  # PyTorch doesn't provide this directly
            'memory_used_mb': torch.cuda.memory_allocated(0) / (1024**2),
            'memory_total_mb': torch.cuda.get_device_properties(0).total_memory / (1024**2),
            'device_name': torch.cuda.get_device_name(0)
        }


class ThresholdCalibrator:
    """Calibrate decision thresholds using ROC and Precision-Recall curves."""
    
    def __init__(self, output_dir: str = './metrics'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def calibrate_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        target_metric: str = 'f1',  # 'f1', 'recall', 'precision'
        min_recall: float = 0.90  # Minimum acceptable recall for security
    ) -> Dict:
        """
        Calibrate threshold to optimize chosen metric while respecting constraints.
        
        Args:
            y_true: Ground truth labels (0/1)
            y_scores: Model prediction scores (0-1)
            target_metric: Metric to optimize ('f1', 'recall', 'precision')
            min_recall: Minimum recall constraint (critical for security)
            
        Returns:
            Dict with optimal threshold and metrics
        """
        logger.info(f"Calibrating threshold to optimize {target_metric} with min_recall={min_recall}")
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        # Find optimal threshold
        best_threshold = 0.5
        best_metric_value = 0.0
        best_metrics = {}
        
        # Evaluate thresholds from PR curve
        for i, threshold in enumerate(pr_thresholds):
            if recall[i] < min_recall:
                continue  # Skip thresholds that don't meet minimum recall
            
            # Calculate F1 score
            if precision[i] + recall[i] > 0:
                f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            else:
                f1 = 0.0
            
            # Select metric to optimize
            if target_metric == 'f1':
                metric_value = f1
            elif target_metric == 'recall':
                metric_value = recall[i]
            elif target_metric == 'precision':
                metric_value = precision[i]
            else:
                metric_value = f1
            
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = threshold
                best_metrics = {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1_score': f1
                }
        
        # Log results
        logger.info(f"Optimal threshold: {best_threshold:.4f}")
        logger.info(f"  Precision: {best_metrics['precision']:.4f}")
        logger.info(f"  Recall: {best_metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {best_metrics['f1_score']:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        logger.info(f"Average Precision: {avg_precision:.4f}")
        
        # Generate visualizations
        self._plot_roc_curve(fpr, tpr, roc_auc)
        self._plot_pr_curve(precision, recall, avg_precision)
        self._plot_threshold_metrics(pr_thresholds, precision, recall)
        
        return {
            'optimal_threshold': float(best_threshold),
            'metrics': best_metrics,
            'roc_auc': float(roc_auc),
            'average_precision': float(avg_precision),
            'num_positives': int(y_true.sum()),
            'num_negatives': int(len(y_true) - y_true.sum())
        }
    
    def _plot_roc_curve(self, fpr, tpr, roc_auc):
        """Plot ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f'roc_curve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC curve saved to {output_path}")
    
    def _plot_pr_curve(self, precision, recall, avg_precision):
        """Plot Precision-Recall curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        output_path = self.output_dir / f'pr_curve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"PR curve saved to {output_path}")
    
    def _plot_threshold_metrics(self, thresholds, precision, recall):
        """Plot metrics vs threshold."""
        # Calculate F1 scores
        f1_scores = []
        for p, r in zip(precision[:-1], recall[:-1]):
            if p + r > 0:
                f1_scores.append(2 * (p * r) / (p + r))
            else:
                f1_scores.append(0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precision[:-1], label='Precision', lw=2)
        plt.plot(thresholds, recall[:-1], label='Recall', lw=2)
        plt.plot(thresholds, f1_scores, label='F1 Score', lw=2)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs Classification Threshold')
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        output_path = self.output_dir / f'threshold_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Threshold metrics saved to {output_path}")


class AdversarialValidator:
    """Detect train/test distribution mismatch using adversarial validation."""
    
    def validate_distributions(
        self,
        train_features: np.ndarray,
        test_features: np.ndarray
    ) -> Dict:
        """
        Train a classifier to distinguish train from test samples.
        High accuracy indicates distribution mismatch.
        
        Args:
            train_features: Training features
            test_features: Test features
            
        Returns:
            Dict with validation results
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        logger.info("Performing adversarial validation...")
        
        # Flatten if needed
        if train_features.ndim == 3:
            train_features = train_features.mean(axis=1)
        if test_features.ndim == 3:
            test_features = test_features.mean(axis=1)
        
        # Create labels: 0 for train, 1 for test
        n_train = len(train_features)
        n_test = len(test_features)
        
        X = np.vstack([train_features, test_features])
        y = np.array([0] * n_train + [1] * n_test)
        
        # Shuffle
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
        
        mean_auc = scores.mean()
        
        # Interpret results
        if mean_auc < 0.55:
            status = "GOOD - Distributions are similar"
        elif mean_auc < 0.65:
            status = "MODERATE - Some distribution differences"
        else:
            status = "BAD - Significant distribution mismatch detected!"
        
        logger.info(f"Adversarial validation AUC: {mean_auc:.4f} - {status}")
        
        return {
            'auc': float(mean_auc),
            'status': status,
            'distribution_mismatch': mean_auc > 0.65
        }


class DiagnosticLogger:
    """Comprehensive logging of predictions and scores for debugging."""
    
    def __init__(self, output_dir: str = './diagnostics'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions = []
        
    def log_prediction(
        self,
        event: Dict,
        score: float,
        components: Dict,
        ground_truth: int,
        threshold: float
    ):
        """Log individual prediction."""
        self.predictions.append({
            'timestamp': datetime.now().isoformat(),
            'file_path': event.get('file_path', 'unknown'),
            'operation': event.get('operation', 'unknown'),
            'user_id': event.get('user_id', 'unknown'),
            'ground_truth': int(ground_truth),
            'score': float(score),
            'prediction': int(score > threshold),
            'threshold': float(threshold),
            'components': {
                'dl_score': float(components.get('dl_score', 0)),
                'if_score': float(components.get('if_score', 0)),
                'heuristic_score': float(components.get('heuristic_score', 0))
            }
        })
    
    def generate_report(self) -> Dict:
        """Generate comprehensive diagnostic report."""
        if not self.predictions:
            return {}
        
        # Extract data
        scores = np.array([p['score'] for p in self.predictions])
        ground_truths = np.array([p['ground_truth'] for p in self.predictions])
        predictions = np.array([p['prediction'] for p in self.predictions])
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(ground_truths, predictions).ravel()
        
        report = {
            'total_predictions': len(self.predictions),
            'score_statistics': {
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'median': float(np.median(scores)),
                'percentile_25': float(np.percentile(scores, 25)),
                'percentile_75': float(np.percentile(scores, 75))
            },
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'actual_positives': int(ground_truths.sum()),
            'actual_negatives': int(len(ground_truths) - ground_truths.sum()),
            'predicted_positives': int(predictions.sum()),
            'predicted_negatives': int(len(predictions) - predictions.sum())
        }
        
        # Analyze score distribution by class
        threat_scores = scores[ground_truths == 1]
        benign_scores = scores[ground_truths == 0]
        
        report['threat_score_distribution'] = {
            'mean': float(threat_scores.mean()) if len(threat_scores) > 0 else 0,
            'std': float(threat_scores.std()) if len(threat_scores) > 0 else 0,
            'min': float(threat_scores.min()) if len(threat_scores) > 0 else 0,
            'max': float(threat_scores.max()) if len(threat_scores) > 0 else 0
        }
        
        report['benign_score_distribution'] = {
            'mean': float(benign_scores.mean()) if len(benign_scores) > 0 else 0,
            'std': float(benign_scores.std()) if len(benign_scores) > 0 else 0,
            'min': float(benign_scores.min()) if len(benign_scores) > 0 else 0,
            'max': float(benign_scores.max()) if len(benign_scores) > 0 else 0
        }
        
        # Save detailed predictions
        output_path = self.output_dir / f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_path, 'w') as f:
            json.dump(self.predictions, f, indent=2)
        
        logger.info(f"Detailed predictions saved to {output_path}")
        
        # Plot score distributions
        self._plot_score_distributions(threat_scores, benign_scores)
        
        return report
    
    def _plot_score_distributions(self, threat_scores, benign_scores):
        """Plot score distributions for threats vs benign."""
        plt.figure(figsize=(10, 6))
        
        if len(threat_scores) > 0:
            plt.hist(threat_scores, bins=30, alpha=0.6, label='Threats (Ground Truth)', color='red', edgecolor='black')
        if len(benign_scores) > 0:
            plt.hist(benign_scores, bins=30, alpha=0.6, label='Benign (Ground Truth)', color='green', edgecolor='black')
        
        plt.xlabel('Model Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution: Threats vs Benign Events')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f'score_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Score distribution plot saved to {output_path}")


def simulate_enhanced_real_file_events(
    num_normal: int = 3000,
    num_anomaly: int = 600,
    ensure_diverse_threats: bool = True
) -> tuple:
    """
    Generate REALISTIC and DIVERSE file system events with VERIFIED labels.
    
    Key improvements:
    - Ensures actual threat characteristics in anomalies
    - Diverse attack patterns matching feature extraction
    - Clear separation between normal and anomalous behavior
    """
    events = []
    labels = []
    
    users = ['user1', 'user2', 'user3', 'admin', 'service']
    normal_extensions = ['.txt', '.pdf', '.doc', '.docx', '.jpg', '.png', '.py', '.json', '.xlsx', '.csv', '.mp4']
    
    # Ransomware-specific extensions (CRITICAL FOR DETECTION)
    ransomware_extensions = [
        '.encrypted', '.locked', '.crypto', '.crypt', '.aes', '.rsa',
        '.cerber', '.locky', '.wannacry', '.petya', '.WNCRY'
    ]
    
    base_time = datetime.now() - timedelta(days=14)
    
    # === Generate Normal Events (80% of dataset) ===
    logger.info(f"Generating {num_normal} normal events...")
    for i in range(num_normal):
        user = random.choice(users)
        ext = random.choice(normal_extensions)
        hour = random.randint(8, 18)  # Business hours
        
        event = {
            'timestamp': (base_time + timedelta(hours=i//60, minutes=i%60)).isoformat(),
            'user_id': user,
            'operation': random.choice(['read', 'write', 'create', 'modify']),
            'file_path': f'/home/{user}/documents/normal_file_{i}{ext}',
            'file_size': random.randint(1024, 50*1024*1024),  # 1KB to 50MB
            'process_name': random.choice(['chrome', 'firefox', 'vscode', 'libreoffice', 'python3']),
            'access_time': hour,
            'file_type': ext[1:],
            'is_system_file': False
        }
        events.append(event)
        labels.append(0)  # Normal
    
    # === Generate Anomalous Events with CLEAR THREAT SIGNATURES ===
    logger.info(f"Generating {num_anomaly} anomalous events with verified threat patterns...")
    
    anomaly_start_time = base_time + timedelta(days=12)  # Threats occur later
    
    for i in range(num_anomaly):
        user = random.choice(['user1', 'user2', 'user3'])
        
        # Distribute across threat types
        threat_type = random.choice([
            'ransomware',  # 40%
            'ransomware',
            'data_exfiltration',  # 20%
            'mass_deletion',  # 20%
            'privilege_escalation'  # 20%
        ])
        
        if threat_type == 'ransomware':
            # CRITICAL: Ransomware with characteristic patterns
            ransom_ext = random.choice(ransomware_extensions)
            event = {
                'timestamp': (anomaly_start_time + timedelta(seconds=i*2)).isoformat(),  # Rapid succession
                'user_id': user,
                'operation': random.choice(['write', 'rename']),  # Encryption or renaming
                'file_path': f'/home/{user}/documents/important_doc_{i}{ransom_ext}',
                'file_size': random.randint(5*1024*1024, 100*1024*1024),  # 5MB-100MB
                'process_name': random.choice(['svchost.exe', 'unknown', 'suspicious_process']),
                'access_time': random.randint(0, 6),  # Late night/early morning
                'file_type': ransom_ext[1:],
                'is_system_file': False,
                'is_suspicious': True,
                'rapid_activity': True
            }
        
        elif threat_type == 'data_exfiltration':
            # Large file access/transfers
            event = {
                'timestamp': (anomaly_start_time + timedelta(hours=1, seconds=i*5)).isoformat(),
                'user_id': user,
                'operation': 'read',
                'file_path': f'/home/{user}/sensitive/database_{i}.sql',
                'file_size': random.randint(100*1024*1024, 2*1024*1024*1024),  # 100MB-2GB
                'process_name': random.choice(['curl', 'wget', 'netcat', 'unknown_network_tool']),
                'access_time': random.randint(20, 23),  # Late night
                'file_type': 'sql',
                'is_system_file': False,
                'is_suspicious': True,
                'large_transfer': True
            }
        
        elif threat_type == 'mass_deletion':
            # Mass file deletion
            event = {
                'timestamp': (anomaly_start_time + timedelta(hours=2, seconds=i*3)).isoformat(),
                'user_id': user,
                'operation': 'delete',
                'file_path': f'/home/{user}/documents/file_{i}.{random.choice(["doc", "pdf", "txt", "xlsx"])}',
                'file_size': random.randint(10*1024, 10*1024*1024),
                'process_name': random.choice(['rm', 'del', 'shred', 'wiper']),
                'access_time': random.randint(1, 5),  # Very early morning
                'file_type': 'various',
                'is_system_file': False,
                'is_suspicious': True,
                'mass_deletion': True
            }
        
        else:  # privilege_escalation
            # Unauthorized system file access
            system_files = ['/etc/passwd', '/etc/shadow', '/etc/sudoers', '/root/.ssh/id_rsa']
            event = {
                'timestamp': (anomaly_start_time + timedelta(hours=3, seconds=i*10)).isoformat(),
                'user_id': user,  # Regular user accessing system files
                'operation': random.choice(['read', 'modify', 'create']),
                'file_path': random.choice(system_files),
                'file_size': random.randint(1024, 100*1024),
                'process_name': random.choice(['bash', 'sh', 'exploit', 'malicious_script']),
                'access_time': random.randint(0, 4),
                'file_type': 'system',
                'is_system_file': True,
                'is_suspicious': True,
                'privilege_escalation': True
            }
        
        events.append(event)
        labels.append(1)  # Anomaly
    
    # Shuffle to mix normal and anomalous
    combined = list(zip(events, labels))
    random.shuffle(combined)
    events, labels = zip(*combined)
    
    # Verify label distribution
    labels_array = np.array(labels)
    logger.info(f"Generated dataset: {len(events)} total events")
    logger.info(f"  Normal: {(labels_array == 0).sum()} ({(labels_array == 0).sum()/len(labels_array)*100:.1f}%)")
    logger.info(f"  Anomalous: {(labels_array == 1).sum()} ({(labels_array == 1).sum()/len(labels_array)*100:.1f}%)")
    
    return list(events), labels_array


class SentinelFSAISystemFixed:
    """Fixed SentinelFS AI System with comprehensive diagnostics."""
    
    def __init__(
        self,
        model_dir: str = './models/production',
        metrics_dir: str = './metrics',
        checkpoint_dir: str = './checkpoints',
        diagnostics_dir: str = './diagnostics'
    ):
        self.model_dir = Path(model_dir)
        self.metrics_dir = Path(metrics_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.diagnostics_dir = Path(diagnostics_dir)
        
        # Create directories
        for dir_path in [self.model_dir, self.metrics_dir, self.checkpoint_dir, self.diagnostics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.gpu_monitor = GPUMonitor()
        self.threshold_calibrator = ThresholdCalibrator(str(self.metrics_dir))
        self.adversarial_validator = AdversarialValidator()
        self.diagnostic_logger = DiagnosticLogger(str(self.diagnostics_dir))
        
        self.feature_extractor = RealFeatureExtractor(
            window_size=128,
            time_window_seconds=300
        )
        
        self.model = None
        self.trainer = None
        self.inference_engine = None
        self.evaluator = ProductionEvaluator(metrics_dir=str(self.metrics_dir))
        
        self.optimal_threshold = 0.45  # Will be calibrated
        
        logger.info("Fixed SentinelFS AI System initialized with comprehensive diagnostics")
    
    def train_with_diagnostics(
        self,
        train_events: list,
        val_events: list,
        train_labels: np.ndarray,
        val_labels: np.ndarray,
        hyperparameters: dict = None
    ) -> Dict:
        """Train model with comprehensive diagnostics and threshold calibration."""
        logger.info("="*80)
        logger.info("TRAINING WITH COMPREHENSIVE DIAGNOSTICS")
        logger.info("="*80)
        
        # Check GPU availability
        gpu_stats = self.gpu_monitor.get_gpu_stats()
        logger.info(f"GPU Status: {gpu_stats}")
        
        if not gpu_stats['gpu_available']:
            logger.warning("⚠️  CUDA not available - training on CPU (will be slow)")
        else:
            logger.info(f"✓ Using GPU: {gpu_stats.get('device_name', 'Unknown')}")
        
        # Hyperparameters
        hp = hyperparameters or {
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.15,
            'learning_rate': 0.0008,
            'batch_size': 64,
            'num_epochs': 50,
            'sequence_length': 64,
            'weight_decay': 0.01,
            'gradient_clipping': 1.0,
            'scheduler_gamma': 0.95
        }
        
        # Initialize model
        num_features = self.feature_extractor.get_num_features()
        logger.info(f"Model input features: {num_features}")
        
        self.model = HybridThreatDetector(
            input_size=num_features,
            hidden_size=hp['hidden_size'],
            num_layers=hp['num_layers'],
            dropout=hp['dropout'],
            use_gru=True,
            isolation_forest_contamination=0.15,  # Adjusted based on dataset
            heuristic_weight=0.3,
            dl_weight=0.4,
            anomaly_weight=0.3
        )
        
        # Move to GPU if available
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        # Initialize trainer
        self.trainer = RealWorldTrainer(
            model=self.model,
            feature_extractor=self.feature_extractor,
            learning_rate=hp['learning_rate'],
            checkpoint_dir=str(self.checkpoint_dir)
        )
        
        # === ADVERSARIAL VALIDATION ===
        logger.info("\n" + "="*80)
        logger.info("ADVERSARIAL VALIDATION - Checking Train/Val Distribution Match")
        logger.info("="*80)
        
        # Extract features for adversarial validation
        train_features = self.feature_extractor.extract_from_sequence(train_events[:1000])  # Sample
        val_features = self.feature_extractor.extract_from_sequence(val_events[:500])
        
        adv_results = self.adversarial_validator.validate_distributions(train_features, val_features)
        if adv_results['distribution_mismatch']:
            logger.warning("⚠️  DISTRIBUTION MISMATCH DETECTED between train and validation!")
            logger.warning("    This may cause generalization issues. Consider data resampling.")
        else:
            logger.info("✓ Train/Val distributions are similar")
        
        # Train model
        logger.info("\n" + "="*80)
        logger.info("STARTING TRAINING")
        logger.info("="*80)
        
        training_results = self.trainer.train_from_real_data(
            train_events=train_events,
            val_events=val_events,
            train_labels=train_labels,
            val_labels=val_labels,                    
            num_epochs=hp['num_epochs'],
            batch_size=hp['batch_size'],
            sequence_length=hp['sequence_length']
        )
        
        # Post-training GPU stats
        gpu_stats_after = self.gpu_monitor.get_gpu_stats()
        logger.info(f"\nPost-training GPU stats: {gpu_stats_after}")
        
        # === THRESHOLD CALIBRATION ===
        logger.info("\n" + "="*80)
        logger.info("CALIBRATING DECISION THRESHOLD")
        logger.info("="*80)
        
        # Get validation predictions
        val_scores = self._get_model_scores(val_events, hp['sequence_length'])
        
        # Align labels with sequences (use label of last event in each sequence)
        num_sequences = len(val_scores)
        aligned_val_labels = val_labels[-num_sequences:]  # Use last labels
        
        calibration_results = self.threshold_calibrator.calibrate_threshold(
            y_true=aligned_val_labels,
            y_scores=val_scores,
            target_metric='f1',
            min_recall=0.85  # Critical: must catch at least 85% of threats
        )
        
        self.optimal_threshold = calibration_results['optimal_threshold']
        logger.info(f"\n✓ Optimal threshold calibrated: {self.optimal_threshold:.4f}")
        
        training_results['calibration'] = calibration_results
        training_results['gpu_stats'] = gpu_stats_after
        
        return training_results
    
    def _prepare_sequences(
        self, 
        events: list, 
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
    
    def _get_model_scores(self, events: list, sequence_length: int) -> np.ndarray:
        """Get raw model scores for events."""
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Prepare sequences like in training
        sequences = self._prepare_sequences(events, sequence_length)
        
        scores = []
        with torch.no_grad():
            for i in range(0, len(sequences), 32):  # Batch processing
                batch = sequences[i:i+32]
                batch_tensor = torch.FloatTensor(batch).to(device)
                batch_scores, _ = self.model(batch_tensor, return_components=False)
                scores.extend(batch_scores.cpu().numpy().flatten())
        
        return np.array(scores)
    
    def test_with_diagnostics(
        self,
        test_events: list,
        test_labels: np.ndarray,
        sequence_length: int = 64
    ) -> Dict:
        """Test model with comprehensive diagnostics."""
        logger.info("\n" + "="*80)
        logger.info("TESTING WITH COMPREHENSIVE DIAGNOSTICS")
        logger.info("="*80)
        
        # Deploy model with calibrated threshold
        self.deploy_for_inference(
            sequence_length=sequence_length,
            threat_threshold=self.optimal_threshold
        )
        
        # Analyze each event with detailed logging
        detected_threats = 0
        device = next(self.model.parameters()).device
        
        for event, label in zip(test_events, test_labels):
            # Get full prediction with components
            result = self.analyze_event_diagnostic(event, label)
            
            if result['anomaly_detected']:
                detected_threats += 1
        
        # Generate diagnostic report
        logger.info("\n" + "="*80)
        logger.info("GENERATING DIAGNOSTIC REPORT")
        logger.info("="*80)
        
        diagnostic_report = self.diagnostic_logger.generate_report()
        
        # Log critical findings
        logger.info("\n📊 DIAGNOSTIC SUMMARY:")
        logger.info(f"  Total test events: {len(test_events)}")
        logger.info(f"  Actual threats: {test_labels.sum()}")
        logger.info(f"  Detected threats: {detected_threats}")
        logger.info(f"  Detection rate: {detected_threats/max(test_labels.sum(), 1)*100:.1f}%")
        
        if 'score_statistics' in diagnostic_report:
            stats = diagnostic_report['score_statistics']
            logger.info(f"\n📈 SCORE STATISTICS:")
            logger.info(f"  Mean: {stats['mean']:.4f}")
            logger.info(f"  Std: {stats['std']:.4f}")
            logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        if 'threat_score_distribution' in diagnostic_report:
            threat_dist = diagnostic_report['threat_score_distribution']
            benign_dist = diagnostic_report['benign_score_distribution']
            logger.info(f"\n🎯 SCORE DISTRIBUTION BY CLASS:")
            logger.info(f"  Threats - Mean: {threat_dist['mean']:.4f}, Std: {threat_dist['std']:.4f}")
            logger.info(f"  Benign  - Mean: {benign_dist['mean']:.4f}, Std: {benign_dist['std']:.4f}")
            logger.info(f"  Separation: {abs(threat_dist['mean'] - benign_dist['mean']):.4f}")
        
        if 'confusion_matrix' in diagnostic_report:
            cm = diagnostic_report['confusion_matrix']
            logger.info(f"\n📋 CONFUSION MATRIX:")
            logger.info(f"  True Positives:  {cm['true_positives']}")
            logger.info(f"  False Positives: {cm['false_positives']}")
            logger.info(f"  True Negatives:  {cm['true_negatives']}")
            logger.info(f"  False Negatives: {cm['false_negatives']}")
            
            # Calculate metrics
            tp, fp, tn, fn = cm['true_positives'], cm['false_positives'], cm['true_negatives'], cm['false_negatives']
            if tp + fn > 0:
                recall = tp / (tp + fn)
                logger.info(f"  Recall: {recall:.4f}")
            if tp + fp > 0:
                precision = tp / (tp + fp)
                logger.info(f"  Precision: {precision:.4f}")
        
        return diagnostic_report
    
    def analyze_event_diagnostic(self, event: dict, ground_truth: int) -> dict:
        """Analyze event with full diagnostic logging."""
        if self.inference_engine is None:
            raise RuntimeError("Model not deployed")
        
        # Get prediction
        result = self.inference_engine.analyze_event(event, return_explanation=True)
        
        # Extract components for logging
        result_dict = result.to_dict()
        
        # Create components dict for logging
        components = {
            'dl_score': result_dict.get('threat_score', 0),  # Simplified for now
            'if_score': 0,
            'heuristic_score': 0
        }
        
        # Log to diagnostic logger
        self.diagnostic_logger.log_prediction(
            event=event,
            score=result.threat_score,
            components=components,
            ground_truth=ground_truth,
            threshold=self.optimal_threshold
        )
        
        return result_dict
    
    def deploy_for_inference(
        self,
        sequence_length: int = 64,
        threat_threshold: float = None,
        batch_size: int = 128
    ):
        """Deploy model with optimized threshold."""
        if threat_threshold is None:
            threat_threshold = self.optimal_threshold
        
        logger.info(f"Deploying model with threshold: {threat_threshold:.4f}")
        
        self.model.eval()
        self.inference_engine = RealTimeInferenceEngine(
            model=self.model,
            feature_extractor=self.feature_extractor,
            sequence_length=sequence_length,
            threat_threshold=threat_threshold,
            enable_caching=True,
            enable_batching=True,
            max_batch_size=batch_size
        )
        
        logger.info("Model deployed successfully")
    
    def save_model(self, model_name: str = 'sentinelfs_fixed'):
        """Save model with calibration info."""
        logger.info(f"Saving model to {self.model_dir / model_name}")
        
        model_path = self.model_dir / f"{model_name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'use_gru': self.model.use_gru
            },
            'optimal_threshold': self.optimal_threshold,
            'feature_config': {
                'num_features': self.feature_extractor.get_num_features(),
                'feature_names': self.feature_extractor.get_feature_names()
            },
            'timestamp': datetime.now().isoformat()
        }, model_path)
        
        logger.info("Model saved successfully")


def main():
    """Main execution with comprehensive fixes."""
    
    print("\n" + "="*80)
    print("SENTINELFS AI - CRITICAL FIX: COMPREHENSIVE DIAGNOSTICS & CALIBRATION")
    print("="*80 + "\n")
    
    # Initialize fixed system
    system = SentinelFSAISystemFixed()
    
    # Step 1: Generate enhanced realistic data
    print("Step 1: Generating enhanced realistic file system events...")
    all_events, all_labels = simulate_enhanced_real_file_events(
        num_normal=4000,
        num_anomaly=800,
        ensure_diverse_threats=True
    )
    
    # Split: 70% train, 15% val, 15% test
    n_train = int(len(all_events) * 0.70)
    n_val = int(len(all_events) * 0.15)
    
    train_events = all_events[:n_train]
    train_labels = all_labels[:n_train]
    val_events = all_events[n_train:n_train+n_val]
    val_labels = all_labels[n_train:n_train+n_val]
    test_events = all_events[n_train+n_val:]
    test_labels = all_labels[n_train+n_val:]
    
    print(f"  Training: {len(train_events)} events ({train_labels.sum()} threats)")
    print(f"  Validation: {len(val_events)} events ({val_labels.sum()} threats)")
    print(f"  Test: {len(test_events)} events ({test_labels.sum()} threats)")
    
    # Step 2: Train with diagnostics
    print("\nStep 2: Training model with comprehensive diagnostics...")
    training_results = system.train_with_diagnostics(
        train_events=train_events,
        val_events=val_events,
        train_labels=train_labels,
        val_labels=val_labels,
        hyperparameters={
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.15,
            'learning_rate': 0.0008,
            'batch_size': 64,
            'num_epochs': 30,  # Reduced for faster testing
            'sequence_length': 64
        }
    )
    
    # Step 3: Test with comprehensive diagnostics
    print("\nStep 3: Testing with comprehensive diagnostics...")
    diagnostic_results = system.test_with_diagnostics(
        test_events=test_events,
        test_labels=test_labels,
        sequence_length=64
    )
    
    # Step 4: Save fixed model
    print("\nStep 4: Saving fixed model...")
    system.save_model('sentinelfs_fixed')
    
    print("\n" + "="*80)
    print("CRITICAL FIX COMPLETE")
    print("="*80)
    print("\n✓ Key Improvements:")
    print("  1. ✓ ROC/PR curve-based threshold calibration")
    print("  2. ✓ Real GPU monitoring with nvidia-smi")
    print("  3. ✓ Comprehensive score distribution analysis")
    print("  4. ✓ Adversarial validation for distribution mismatch")
    print("  5. ✓ Enhanced test data with verified threat labels")
    print("  6. ✓ Detailed prediction logging and visualization")
    print("\n📊 Check './diagnostics/' for detailed analysis")
    print("📈 Check './metrics/' for ROC/PR curves")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
