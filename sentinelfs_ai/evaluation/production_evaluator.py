"""
Production evaluation and monitoring system for model performance tracking.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ProductionEvaluator:
    """
    Real-world model evaluation and monitoring system.
    
    Features:
    - Continuous performance tracking
    - False positive/negative analysis
    - Model drift detection
    - A/B testing support
    - Alerting thresholds
    - Exportable metrics
    """
    
    def __init__(
        self,
        metrics_dir: str = './metrics',
        window_size: int = 1000,
        drift_threshold: float = 0.1,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize production evaluator.
        
        Args:
            metrics_dir: Directory to store metrics
            window_size: Rolling window size for metrics
            drift_threshold: Threshold for drift detection
            alert_thresholds: Custom alert thresholds
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'false_positive_rate': 0.05,  # 5%
            'false_negative_rate': 0.10,  # 10%
            'accuracy': 0.85,  # 85%
            'f1_score': 0.80,  # 80%
            'latency_p99_ms': 25.0,  # 25ms
            'threat_detection_rate': 0.01  # 1%
        }
        
        # Metrics storage
        self.predictions: deque = deque(maxlen=window_size)
        self.ground_truth: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)
        self.latencies: deque = deque(maxlen=window_size)
        self.scores: deque = deque(maxlen=window_size)
        
        # Detailed tracking
        self.false_positives: List[Dict[str, Any]] = []
        self.false_negatives: List[Dict[str, Any]] = []
        self.true_positives: List[Dict[str, Any]] = []
        self.true_negatives: List[Dict[str, Any]] = []
        
        # Drift detection
        self.baseline_metrics: Optional[Dict[str, float]] = None
        self.drift_alerts: List[Dict[str, Any]] = []
        
        # Performance over time
        self.daily_metrics: Dict[str, List[float]] = defaultdict(list)
        
        logger.info("Production evaluator initialized")
    
    def record_prediction(
        self,
        prediction: float,
        ground_truth: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None
    ):
        """
        Record a prediction for evaluation.
        
        Args:
            prediction: Model prediction score (0-1)
            ground_truth: True label (0 or 1), if available
            metadata: Additional context about the prediction
            latency_ms: Inference latency
        """
        timestamp = datetime.now()
        
        self.predictions.append(prediction)
        self.timestamps.append(timestamp)
        self.scores.append(prediction)
        
        if ground_truth is not None:
            self.ground_truth.append(ground_truth)
            
            # Classify prediction
            predicted_label = 1 if prediction > 0.5 else 0
            
            record = {
                'timestamp': timestamp.isoformat(),
                'prediction': prediction,
                'ground_truth': ground_truth,
                'metadata': metadata or {}
            }
            
            if predicted_label == 1 and ground_truth == 1:
                self.true_positives.append(record)
            elif predicted_label == 1 and ground_truth == 0:
                self.false_positives.append(record)
            elif predicted_label == 0 and ground_truth == 1:
                self.false_negatives.append(record)
            else:
                self.true_negatives.append(record)
        
        if latency_ms is not None:
            self.latencies.append(latency_ms)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate current performance metrics."""
        if len(self.ground_truth) == 0:
            return {'error': 'No ground truth labels available'}
        
        # Align predictions and ground truth
        n = min(len(self.predictions), len(self.ground_truth))
        preds = np.array(list(self.predictions)[-n:])
        truth = np.array(list(self.ground_truth)[-n:])
        
        # Convert predictions to labels
        pred_labels = (preds > 0.5).astype(int)
        
        # Confusion matrix
        tp = np.sum((pred_labels == 1) & (truth == 1))
        tn = np.sum((pred_labels == 0) & (truth == 0))
        fp = np.sum((pred_labels == 1) & (truth == 0))
        fn = np.sum((pred_labels == 0) & (truth == 1))
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # AUC-ROC approximation
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(truth, preds)
        except:
            auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'specificity': specificity,
            'auc_roc': auc,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': int(len(truth))
        }
        
        # Latency metrics
        if len(self.latencies) > 0:
            latencies = np.array(list(self.latencies))
            metrics.update({
                'latency_mean_ms': float(np.mean(latencies)),
                'latency_median_ms': float(np.median(latencies)),
                'latency_p95_ms': float(np.percentile(latencies, 95)),
                'latency_p99_ms': float(np.percentile(latencies, 99)),
                'latency_max_ms': float(np.max(latencies))
            })
        
        return metrics
    
    def detect_drift(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect model drift by comparing current metrics to baseline.
        
        Returns:
            (drift_detected, drift_report)
        """
        current_metrics = self.calculate_metrics()
        
        if self.baseline_metrics is None:
            # Set current as baseline
            self.baseline_metrics = current_metrics
            logger.info("Baseline metrics set")
            return False, {'status': 'baseline_set'}
        
        # Compare key metrics
        drift_detected = False
        drifted_metrics = {}
        
        key_metrics = ['accuracy', 'f1_score', 'false_positive_rate', 'false_negative_rate']
        
        for metric in key_metrics:
            if metric in current_metrics and metric in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric]
                current_value = current_metrics[metric]
                
                # Calculate relative change
                if baseline_value != 0:
                    change = abs(current_value - baseline_value) / baseline_value
                else:
                    change = abs(current_value - baseline_value)
                
                if change > self.drift_threshold:
                    drift_detected = True
                    drifted_metrics[metric] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'change': change,
                        'threshold': self.drift_threshold
                    }
        
        drift_report = {
            'drift_detected': drift_detected,
            'timestamp': datetime.now().isoformat(),
            'drifted_metrics': drifted_metrics,
            'current_metrics': current_metrics,
            'baseline_metrics': self.baseline_metrics
        }
        
        if drift_detected:
            self.drift_alerts.append(drift_report)
            logger.warning(f"Model drift detected! Drifted metrics: {list(drifted_metrics.keys())}")
        
        return drift_detected, drift_report
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check if any metrics exceed alert thresholds."""
        current_metrics = self.calculate_metrics()
        alerts = []
        
        for metric, threshold in self.alert_thresholds.items():
            if metric not in current_metrics:
                continue
            
            value = current_metrics[metric]
            
            # Determine if threshold is exceeded
            violated = False
            if metric in ['false_positive_rate', 'false_negative_rate', 'latency_p99_ms']:
                # Lower is better
                violated = value > threshold
            else:
                # Higher is better
                violated = value < threshold
            
            if violated:
                alert = {
                    'metric': metric,
                    'value': value,
                    'threshold': threshold,
                    'severity': 'high' if abs(value - threshold) / threshold > 0.2 else 'medium',
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
                logger.warning(f"Alert: {metric} = {value:.4f} (threshold: {threshold})")
        
        return alerts
    
    def analyze_false_positives(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Analyze most recent false positives for patterns."""
        if len(self.false_positives) == 0:
            return []
        
        # Get recent false positives
        recent_fps = self.false_positives[-top_k:]
        
        # Analyze patterns
        analysis = []
        for fp in recent_fps:
            metadata = fp.get('metadata', {})
            analysis.append({
                'timestamp': fp['timestamp'],
                'prediction_score': fp['prediction'],
                'file_path': metadata.get('file_path', 'unknown'),
                'operation': metadata.get('operation', 'unknown'),
                'user_id': metadata.get('user_id', 'unknown'),
                'explanation': metadata.get('explanation', {})
            })
        
        return analysis
    
    def analyze_false_negatives(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Analyze most recent false negatives for missed threats."""
        if len(self.false_negatives) == 0:
            return []
        
        # Get recent false negatives
        recent_fns = self.false_negatives[-top_k:]
        
        # Analyze patterns
        analysis = []
        for fn in recent_fns:
            metadata = fn.get('metadata', {})
            analysis.append({
                'timestamp': fn['timestamp'],
                'prediction_score': fn['prediction'],
                'file_path': metadata.get('file_path', 'unknown'),
                'operation': metadata.get('operation', 'unknown'),
                'user_id': metadata.get('user_id', 'unknown'),
                'actual_threat_type': metadata.get('threat_type', 'unknown')
            })
        
        return analysis
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        metrics = self.calculate_metrics()
        alerts = self.check_alerts()
        drift_detected, drift_report = self.detect_drift()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_predictions': len(self.predictions),
                'labeled_samples': len(self.ground_truth),
                'evaluation_window': self.window_size
            },
            'performance_metrics': metrics,
            'drift_detection': drift_report,
            'active_alerts': alerts,
            'false_positive_analysis': self.analyze_false_positives(),
            'false_negative_analysis': self.analyze_false_negatives(),
            'recommendations': self._generate_recommendations(metrics, alerts, drift_detected)
        }
        
        # Save report
        report_path = self.metrics_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        return report
    
    def _generate_recommendations(
        self,
        metrics: Dict[str, float],
        alerts: List[Dict[str, Any]],
        drift_detected: bool
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Check false positive rate
        if metrics.get('false_positive_rate', 0) > 0.1:
            recommendations.append(
                "High false positive rate detected. Consider:\n"
                "  1. Increasing detection threshold\n"
                "  2. Retraining with more normal behavior examples\n"
                "  3. Adjusting heuristic rules"
            )
        
        # Check false negative rate
        if metrics.get('false_negative_rate', 0) > 0.15:
            recommendations.append(
                "High false negative rate detected. Consider:\n"
                "  1. Decreasing detection threshold\n"
                "  2. Adding more threat examples to training\n"
                "  3. Enhancing feature extraction"
            )
        
        # Check latency
        if metrics.get('latency_p99_ms', 0) > 25:
            recommendations.append(
                "Latency exceeds target (<25ms). Consider:\n"
                "  1. Using lightweight model variant\n"
                "  2. Optimizing feature extraction\n"
                "  3. Enabling batching and caching"
            )
        
        # Check drift
        if drift_detected:
            recommendations.append(
                "Model drift detected. Immediate actions:\n"
                "  1. Collect recent data for retraining\n"
                "  2. Perform incremental learning update\n"
                "  3. Review and update heuristic rules\n"
                "  4. Schedule full model retraining"
            )
        
        # Check overall performance
        if metrics.get('f1_score', 0) < 0.80:
            recommendations.append(
                "Overall performance below target. Consider:\n"
                "  1. Full model retraining with expanded dataset\n"
                "  2. Hyperparameter tuning\n"
                "  3. Ensemble model approach\n"
                "  4. Feature engineering improvements"
            )
        
        return recommendations if recommendations else ["Model performing within acceptable parameters"]
    
    def export_metrics_for_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        metrics = self.calculate_metrics()
        
        prometheus_metrics = []
        
        # Accuracy
        prometheus_metrics.append(
            f"sentinel_ai_accuracy {metrics.get('accuracy', 0):.6f}"
        )
        
        # Precision
        prometheus_metrics.append(
            f"sentinel_ai_precision {metrics.get('precision', 0):.6f}"
        )
        
        # Recall
        prometheus_metrics.append(
            f"sentinel_ai_recall {metrics.get('recall', 0):.6f}"
        )
        
        # F1 Score
        prometheus_metrics.append(
            f"sentinel_ai_f1_score {metrics.get('f1_score', 0):.6f}"
        )
        
        # False Positive Rate
        prometheus_metrics.append(
            f"sentinel_ai_false_positive_rate {metrics.get('false_positive_rate', 0):.6f}"
        )
        
        # False Negative Rate
        prometheus_metrics.append(
            f"sentinel_ai_false_negative_rate {metrics.get('false_negative_rate', 0):.6f}"
        )
        
        # Latency
        if 'latency_p99_ms' in metrics:
            prometheus_metrics.append(
                f"sentinel_ai_latency_p99_ms {metrics['latency_p99_ms']:.2f}"
            )
        
        # Confusion matrix
        prometheus_metrics.append(
            f"sentinel_ai_true_positives {metrics.get('true_positives', 0)}"
        )
        prometheus_metrics.append(
            f"sentinel_ai_false_positives {metrics.get('false_positives', 0)}"
        )
        prometheus_metrics.append(
            f"sentinel_ai_true_negatives {metrics.get('true_negatives', 0)}"
        )
        prometheus_metrics.append(
            f"sentinel_ai_false_negatives {metrics.get('false_negatives', 0)}"
        )
        
        return '\n'.join(prometheus_metrics)
    
    def reset_baseline(self):
        """Reset baseline metrics for drift detection."""
        self.baseline_metrics = None
        self.drift_alerts.clear()
        logger.info("Baseline metrics reset")
    
    def save_state(self):
        """Save evaluator state to disk."""
        state = {
            'baseline_metrics': self.baseline_metrics,
            'drift_alerts': self.drift_alerts,
            'false_positive_count': len(self.false_positives),
            'false_negative_count': len(self.false_negatives),
            'total_predictions': len(self.predictions),
            'timestamp': datetime.now().isoformat()
        }
        
        state_path = self.metrics_dir / 'evaluator_state.json'
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Evaluator state saved to {state_path}")
    
    def load_state(self, state_path: str):
        """Load evaluator state from disk."""
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        self.baseline_metrics = state.get('baseline_metrics')
        self.drift_alerts = state.get('drift_alerts', [])
        
        logger.info(f"Evaluator state loaded from {state_path}")
