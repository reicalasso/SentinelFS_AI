"""
Security Validation Framework
=============================

Continuous security monitoring and validation for production deployments.
Monitors for adversarial attacks, drift, and anomalies in real-time.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import logging
import time
import json
from datetime import datetime

from .attack_generator import AttackGenerator, AttackConfig
from .defense_mechanisms import AdversarialDetector, DefenseConfig
from .robustness_tester import RobustnessTester

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for security validation"""
    
    # Monitoring parameters
    window_size: int = 1000  # Samples to monitor
    validation_interval: int = 100  # Validate every N samples
    
    # Thresholds
    confidence_threshold: float = 0.9  # Minimum acceptable confidence
    accuracy_threshold: float = 0.95  # Minimum acceptable accuracy
    anomaly_threshold: float = 0.05  # Maximum anomaly rate
    
    # Attack detection
    enable_attack_detection: bool = True
    detection_sensitivity: float = 0.7
    
    # Periodic testing
    enable_periodic_testing: bool = True
    test_frequency: int = 1000  # Test every N samples
    test_attack_types: List[str] = field(
        default_factory=lambda: ['fgsm', 'pgd']
    )
    
    # Alerting
    enable_alerts: bool = True
    alert_cooldown: int = 300  # Seconds between alerts
    
    # Logging
    log_file: Optional[str] = "security_validation.log"
    save_interval: int = 100


class SecurityEvent:
    """Container for security events"""
    
    def __init__(
        self,
        event_type: str,
        severity: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.timestamp = datetime.now()
        self.event_type = event_type
        self.severity = severity  # 'info', 'warning', 'critical'
        self.message = message
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'severity': self.severity,
            'message': self.message,
            'metadata': self.metadata
        }


class SecurityValidator:
    """
    Security Validation Manager
    
    Continuously monitors model for security issues:
    - Adversarial attack detection
    - Anomaly detection
    - Performance degradation
    - Periodic robustness testing
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[ValidationConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize security validator
        
        Args:
            model: Model to monitor
            config: Validation configuration
            device: Device to use
        """
        self.model = model
        self.config = config or ValidationConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize detectors
        defense_config = DefenseConfig(
            use_detection=True,
            detection_threshold=self.config.detection_sensitivity
        )
        self.detector = AdversarialDetector(model, defense_config)
        
        # Initialize tester for periodic validation
        if self.config.enable_periodic_testing:
            attack_config = AttackConfig(epsilon=0.3, num_steps=20)
            self.tester = RobustnessTester(
                model, attack_config, defense_config, device
            )
        
        # Monitoring state
        self.predictions_window = deque(maxlen=self.config.window_size)
        self.confidences_window = deque(maxlen=self.config.window_size)
        self.anomaly_scores_window = deque(maxlen=self.config.window_size)
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'correct_predictions': 0,
            'detected_attacks': 0,
            'anomalies': 0,
            'low_confidence_predictions': 0,
            'validation_runs': 0
        }
        
        # Security events
        self.events: List[SecurityEvent] = []
        self.last_alert_time = 0
        
        # Periodic test results
        self.test_history = []
        
        logger.info("Security Validator initialized")
        logger.info(f"Window size: {self.config.window_size}")
        logger.info(f"Validation interval: {self.config.validation_interval}")
    
    def validate_sample(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Validate a single sample or batch
        
        Args:
            x: Input tensor [batch_size, features]
            y: Optional true labels [batch_size]
            return_details: Whether to return detailed results
            
        Returns:
            Validation results
        """
        x = x.to(self.device)
        if y is not None:
            y = y.to(self.device)
        
        batch_size = x.shape[0]
        results = {
            'is_safe': True,
            'warnings': [],
            'detections': {}
        }
        
        # 1. Adversarial detection
        if self.config.enable_attack_detection:
            detection_scores, is_adversarial = self.detector.detect(x)
            num_detected = is_adversarial.sum().item()
            
            if num_detected > 0:
                results['is_safe'] = False
                results['warnings'].append(
                    f"Detected {num_detected}/{batch_size} potential adversarial examples"
                )
                results['detections']['adversarial'] = {
                    'count': num_detected,
                    'scores': detection_scores.cpu().numpy().tolist(),
                    'indices': is_adversarial.nonzero().squeeze().cpu().numpy().tolist()
                }
                
                self.stats['detected_attacks'] += num_detected
                
                # Log security event
                self._log_event(
                    'adversarial_detection',
                    'warning',
                    f"Detected {num_detected} adversarial examples",
                    {'detection_scores': detection_scores.mean().item()}
                )
        
        # 2. Model predictions and confidence
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            probs = torch.softmax(outputs, dim=1)
            confidence, predictions = probs.max(dim=1)
            
            # Check confidence
            low_conf = confidence < self.config.confidence_threshold
            num_low_conf = low_conf.sum().item()
            
            if num_low_conf > 0:
                results['warnings'].append(
                    f"{num_low_conf}/{batch_size} predictions have low confidence"
                )
                results['detections']['low_confidence'] = {
                    'count': num_low_conf,
                    'scores': confidence[low_conf].cpu().numpy().tolist(),
                    'indices': low_conf.nonzero().squeeze().cpu().numpy().tolist()
                }
                
                self.stats['low_confidence_predictions'] += num_low_conf
            
            # Update windows
            self.predictions_window.extend(predictions.cpu().numpy())
            self.confidences_window.extend(confidence.cpu().numpy())
            
            # Check accuracy if labels provided
            if y is not None:
                correct = predictions.eq(y).sum().item()
                accuracy = correct / batch_size
                
                self.stats['correct_predictions'] += correct
                
                if accuracy < self.config.accuracy_threshold:
                    results['is_safe'] = False
                    results['warnings'].append(
                        f"Batch accuracy {accuracy:.2%} below threshold"
                    )
        
        # 3. Anomaly detection (using confidence distribution)
        if len(self.confidences_window) >= 100:
            current_conf = confidence.mean().item()
            historical_mean = np.mean(list(self.confidences_window)[:-batch_size])
            historical_std = np.std(list(self.confidences_window)[:-batch_size])
            
            # Z-score based anomaly detection
            if historical_std > 0:
                z_score = abs(current_conf - historical_mean) / historical_std
                
                if z_score > 3:  # 3-sigma rule
                    results['warnings'].append(
                        f"Anomalous confidence distribution (z-score: {z_score:.2f})"
                    )
                    self.stats['anomalies'] += 1
                    
                    self._log_event(
                        'anomaly_detection',
                        'warning',
                        f"Confidence anomaly detected (z-score: {z_score:.2f})",
                        {'z_score': z_score, 'current_confidence': current_conf}
                    )
        
        # Update statistics
        self.stats['total_samples'] += batch_size
        
        # Periodic validation
        if (self.stats['total_samples'] % self.config.validation_interval == 0):
            self._periodic_validation()
        
        if return_details:
            results['statistics'] = self.get_statistics()
            results['confidence_stats'] = {
                'mean': confidence.mean().item(),
                'std': confidence.std().item(),
                'min': confidence.min().item(),
                'max': confidence.max().item()
            }
        
        return results
    
    def _periodic_validation(self):
        """Run periodic comprehensive validation"""
        self.stats['validation_runs'] += 1
        
        logger.info(
            f"Periodic validation #{self.stats['validation_runs']} "
            f"(samples: {self.stats['total_samples']})"
        )
        
        # Calculate rolling statistics
        if len(self.confidences_window) > 0:
            avg_confidence = np.mean(list(self.confidences_window))
            
            # Check if confidence is degrading
            if avg_confidence < self.config.confidence_threshold:
                self._log_event(
                    'performance_degradation',
                    'warning',
                    f"Average confidence dropped to {avg_confidence:.4f}",
                    {'avg_confidence': avg_confidence}
                )
        
        # Check attack detection rate
        if self.stats['total_samples'] > 0:
            attack_rate = self.stats['detected_attacks'] / self.stats['total_samples']
            
            if attack_rate > self.config.anomaly_threshold:
                self._log_event(
                    'high_attack_rate',
                    'critical',
                    f"High attack detection rate: {attack_rate:.2%}",
                    {'attack_rate': attack_rate}
                )
                
                # Trigger alert
                self._trigger_alert(
                    f"Critical: Attack detection rate at {attack_rate:.2%}"
                )
    
    def run_security_test(
        self,
        test_loader,
        attack_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive security test
        
        Args:
            test_loader: Test data loader
            attack_types: List of attacks to test
            
        Returns:
            Test results
        """
        if not self.config.enable_periodic_testing:
            logger.warning("Periodic testing is disabled")
            return {}
        
        logger.info("Running comprehensive security test...")
        
        attack_types = attack_types or self.config.test_attack_types
        
        # Run robustness test
        metrics = self.tester.test_comprehensive(
            test_loader,
            attack_types,
            use_defense=True
        )
        
        # Store results
        test_result = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics.to_dict()
        }
        self.test_history.append(test_result)
        
        # Check for degradation
        if metrics.robust_accuracy < 0.7:
            self._log_event(
                'robustness_degradation',
                'critical',
                f"Robust accuracy dropped to {metrics.robust_accuracy:.2%}",
                {'robust_accuracy': metrics.robust_accuracy}
            )
            
            self._trigger_alert(
                f"Critical: Robust accuracy at {metrics.robust_accuracy:.2%}"
            )
        
        logger.info(f"Security test completed. Robust accuracy: {metrics.robust_accuracy:.2%}")
        
        return test_result
    
    def calibrate(self, clean_data_loader):
        """
        Calibrate detector on clean data
        
        Args:
            clean_data_loader: DataLoader with clean examples
        """
        logger.info("Calibrating security validator...")
        self.detector.calibrate(clean_data_loader)
        logger.info("Calibration complete")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current monitoring statistics"""
        stats = self.stats.copy()
        
        # Calculate derived metrics
        if stats['total_samples'] > 0:
            stats['accuracy'] = stats['correct_predictions'] / stats['total_samples']
            stats['attack_detection_rate'] = stats['detected_attacks'] / stats['total_samples']
            stats['anomaly_rate'] = stats['anomalies'] / stats['total_samples']
        else:
            stats['accuracy'] = 0.0
            stats['attack_detection_rate'] = 0.0
            stats['anomaly_rate'] = 0.0
        
        if len(self.confidences_window) > 0:
            stats['avg_confidence'] = np.mean(list(self.confidences_window))
            stats['confidence_std'] = np.std(list(self.confidences_window))
        else:
            stats['avg_confidence'] = 0.0
            stats['confidence_std'] = 0.0
        
        return stats
    
    def get_events(
        self,
        severity: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get security events
        
        Args:
            severity: Filter by severity
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        events = self.events
        
        if severity:
            events = [e for e in events if e.severity == severity]
        
        if limit:
            events = events[-limit:]
        
        return [e.to_dict() for e in events]
    
    def _log_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a security event"""
        event = SecurityEvent(event_type, severity, message, metadata)
        self.events.append(event)
        
        # Log to logger
        log_func = {
            'info': logger.info,
            'warning': logger.warning,
            'critical': logger.critical
        }.get(severity, logger.info)
        
        log_func(f"[{event_type}] {message}")
        
        # Save to file if configured
        if self.config.log_file:
            self._save_event_to_file(event)
    
    def _save_event_to_file(self, event: SecurityEvent):
        """Save event to log file"""
        try:
            with open(self.config.log_file, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Error saving event to file: {e}")
    
    def _trigger_alert(self, message: str):
        """Trigger alert (with cooldown)"""
        if not self.config.enable_alerts:
            return
        
        current_time = time.time()
        
        if current_time - self.last_alert_time < self.config.alert_cooldown:
            logger.debug("Alert suppressed (cooldown active)")
            return
        
        logger.critical(f"🚨 SECURITY ALERT: {message}")
        
        self.last_alert_time = current_time
    
    def generate_security_report(self, filepath: str):
        """Generate comprehensive security report"""
        stats = self.get_statistics()
        
        report_lines = [
            "# Security Validation Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Monitoring Statistics",
            "",
            f"- **Total Samples Processed**: {stats['total_samples']}",
            f"- **Overall Accuracy**: {stats.get('accuracy', 0):.2%}",
            f"- **Average Confidence**: {stats.get('avg_confidence', 0):.4f}",
            f"- **Attack Detection Rate**: {stats.get('attack_detection_rate', 0):.2%}",
            f"- **Anomaly Rate**: {stats.get('anomaly_rate', 0):.2%}",
            "",
            "## Security Events",
            ""
        ]
        
        # Group events by type
        events_by_type = {}
        for event in self.events:
            event_type = event.event_type
            if event_type not in events_by_type:
                events_by_type[event_type] = []
            events_by_type[event_type].append(event)
        
        for event_type, events in events_by_type.items():
            report_lines.append(f"### {event_type.replace('_', ' ').title()}")
            report_lines.append(f"Count: {len(events)}")
            
            # Show recent events
            recent = events[-5:]
            if recent:
                report_lines.append("\nRecent Events:")
                for event in recent:
                    report_lines.append(
                        f"- [{event.severity.upper()}] {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: "
                        f"{event.message}"
                    )
            report_lines.append("")
        
        # Test history
        if self.test_history:
            report_lines.extend([
                "## Robustness Test History",
                ""
            ])
            
            for i, test in enumerate(self.test_history[-5:], 1):
                metrics = test['metrics']
                report_lines.extend([
                    f"### Test #{i} - {test['timestamp']}",
                    f"- Clean Accuracy: {metrics['clean_accuracy']:.2%}",
                    f"- Robust Accuracy: {metrics['robust_accuracy']:.2%}",
                    ""
                ])
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            "",
            self._generate_security_recommendations(stats),
            ""
        ])
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Security report saved to {filepath}")
    
    def _generate_security_recommendations(self, stats: Dict[str, Any]) -> str:
        """Generate security recommendations"""
        recommendations = []
        
        attack_rate = stats.get('attack_detection_rate', 0)
        if attack_rate > 0.1:
            recommendations.append(
                f"⚠️ **High attack rate** ({attack_rate:.2%}): "
                "Consider enabling additional defenses or investigating the traffic source."
            )
        
        accuracy = stats.get('accuracy', 0)
        if accuracy < 0.9:
            recommendations.append(
                f"⚠️ **Low accuracy** ({accuracy:.2%}): "
                "Model performance may be degraded. Consider retraining or investigating data drift."
            )
        
        avg_conf = stats.get('avg_confidence', 0)
        if avg_conf < 0.8:
            recommendations.append(
                f"⚠️ **Low confidence** ({avg_conf:.4f}): "
                "Model may be uncertain. Consider additional validation."
            )
        
        if not recommendations:
            recommendations.append(
                "✅ **No critical issues detected**. Continue monitoring."
            )
        
        return '\n'.join(f"- {r}" for r in recommendations)
