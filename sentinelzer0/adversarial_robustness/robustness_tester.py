"""
Robustness Testing Suite
========================

Comprehensive framework for testing model robustness against adversarial attacks.
Provides detailed metrics, visualization, and reporting capabilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging
import time
import json

from .attack_generator import AttackGenerator, AttackConfig
from .defense_mechanisms import DefenseManager, DefenseConfig

logger = logging.getLogger(__name__)


@dataclass
class RobustnessMetrics:
    """Container for robustness metrics"""
    
    # Accuracy metrics
    clean_accuracy: float = 0.0
    adversarial_accuracy: Dict[str, float] = field(default_factory=dict)
    robust_accuracy: float = 0.0  # Minimum across all attacks
    
    # Attack success rates
    attack_success_rate: Dict[str, float] = field(default_factory=dict)
    
    # Perturbation metrics
    l2_distance: Dict[str, float] = field(default_factory=dict)
    linf_distance: Dict[str, float] = field(default_factory=dict)
    
    # Confidence metrics
    clean_confidence: float = 0.0
    adversarial_confidence: Dict[str, float] = field(default_factory=dict)
    
    # Timing
    attack_time: Dict[str, float] = field(default_factory=dict)
    defense_time: float = 0.0
    
    # Summary statistics
    total_samples: int = 0
    successful_attacks: int = 0
    detected_attacks: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'clean_accuracy': self.clean_accuracy,
            'adversarial_accuracy': self.adversarial_accuracy,
            'robust_accuracy': self.robust_accuracy,
            'attack_success_rate': self.attack_success_rate,
            'l2_distance': self.l2_distance,
            'linf_distance': self.linf_distance,
            'clean_confidence': self.clean_confidence,
            'adversarial_confidence': self.adversarial_confidence,
            'attack_time': self.attack_time,
            'defense_time': self.defense_time,
            'total_samples': self.total_samples,
            'successful_attacks': self.successful_attacks,
            'detected_attacks': self.detected_attacks
        }
    
    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            "=" * 60,
            "ROBUSTNESS TEST SUMMARY",
            "=" * 60,
            f"Total Samples: {self.total_samples}",
            f"Clean Accuracy: {self.clean_accuracy:.2%}",
            f"Robust Accuracy: {self.robust_accuracy:.2%}",
            "",
            "Attack Results:",
            "-" * 60
        ]
        
        for attack_type in self.adversarial_accuracy:
            lines.extend([
                f"{attack_type.upper()}:",
                f"  Accuracy: {self.adversarial_accuracy[attack_type]:.2%}",
                f"  Success Rate: {self.attack_success_rate[attack_type]:.2%}",
                f"  L2 Distance: {self.l2_distance.get(attack_type, 0):.4f}",
                f"  L∞ Distance: {self.linf_distance.get(attack_type, 0):.4f}",
                f"  Avg Confidence: {self.adversarial_confidence.get(attack_type, 0):.4f}",
                f"  Time: {self.attack_time.get(attack_type, 0):.2f}s",
                ""
            ])
        
        lines.extend([
            "=" * 60,
            f"Successful Attacks: {self.successful_attacks}/{self.total_samples}",
            f"Detected Attacks: {self.detected_attacks}",
            "=" * 60
        ])
        
        return "\n".join(lines)


class RobustnessTester:
    """
    Comprehensive Robustness Testing Framework
    
    Tests model robustness against multiple adversarial attacks and
    evaluates defense mechanisms.
    """
    
    def __init__(
        self,
        model: nn.Module,
        attack_config: Optional[AttackConfig] = None,
        defense_config: Optional[DefenseConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize robustness tester
        
        Args:
            model: Model to test
            attack_config: Configuration for attacks
            defense_config: Configuration for defenses
            device: Device to use
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize attack generator
        self.attack_config = attack_config or AttackConfig()
        self.attack_generator = AttackGenerator(model, self.attack_config)
        
        # Initialize defense manager
        self.defense_config = defense_config or DefenseConfig()
        self.defense_manager = DefenseManager(model, self.defense_config)
        
        self.metrics = RobustnessMetrics()
        
    def test_clean_accuracy(
        self,
        data_loader: DataLoader
    ) -> Tuple[float, float]:
        """
        Test clean accuracy (no attacks)
        
        Args:
            data_loader: Test data loader
            
        Returns:
            Tuple of (accuracy, average confidence)
        """
        self.model.eval()
        correct = 0
        total = 0
        confidences = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = self.model(data)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Predictions
                probs = torch.softmax(outputs, dim=1)
                pred = outputs.argmax(dim=1)
                
                # Statistics
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                confidences.extend(probs.max(dim=1)[0].cpu().numpy())
        
        accuracy = correct / total if total > 0 else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return accuracy, avg_confidence
    
    def test_adversarial_attack(
        self,
        data_loader: DataLoader,
        attack_type: str = 'pgd',
        use_defense: bool = False
    ) -> Dict[str, float]:
        """
        Test robustness against a specific attack
        
        Args:
            data_loader: Test data loader
            attack_type: Type of attack to test
            use_defense: Whether to apply defenses
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        correct = 0
        total = 0
        confidences = []
        l2_distances = []
        linf_distances = []
        
        start_time = time.time()
        
        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            # Generate adversarial examples
            data_adv = self.attack_generator.generate(
                data, target, attack_type
            )
            
            # Apply defenses if requested
            if use_defense:
                data_adv, _ = self.defense_manager.defend(data_adv)
            
            # Evaluate
            with torch.no_grad():
                outputs = self.model(data_adv)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Predictions
                probs = torch.softmax(outputs, dim=1)
                pred = outputs.argmax(dim=1)
                
                # Statistics
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                confidences.extend(probs.max(dim=1)[0].cpu().numpy())
                
                # Perturbation metrics
                l2_dist = torch.norm(data_adv - data, p=2, dim=1)
                linf_dist = torch.norm(data_adv - data, p=float('inf'), dim=1)
                l2_distances.extend(l2_dist.cpu().numpy())
                linf_distances.extend(linf_dist.cpu().numpy())
        
        attack_time = time.time() - start_time
        
        accuracy = correct / total if total > 0 else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        avg_l2 = np.mean(l2_distances) if l2_distances else 0.0
        avg_linf = np.mean(linf_distances) if linf_distances else 0.0
        
        return {
            'accuracy': accuracy,
            'confidence': avg_confidence,
            'l2_distance': avg_l2,
            'linf_distance': avg_linf,
            'time': attack_time
        }
    
    def test_comprehensive(
        self,
        data_loader: DataLoader,
        attack_types: Optional[List[str]] = None,
        use_defense: bool = False
    ) -> RobustnessMetrics:
        """
        Run comprehensive robustness tests
        
        Args:
            data_loader: Test data loader
            attack_types: List of attacks to test
            use_defense: Whether to apply defenses
            
        Returns:
            Comprehensive robustness metrics
        """
        logger.info("Starting comprehensive robustness testing...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Defense: {'Enabled' if use_defense else 'Disabled'}")
        
        if attack_types is None:
            attack_types = ['fgsm', 'pgd', 'cw', 'boundary']
        
        # Calculate total samples
        total_samples = len(data_loader.dataset)
        self.metrics.total_samples = total_samples
        
        # Test clean accuracy
        logger.info("Testing clean accuracy...")
        clean_acc, clean_conf = self.test_clean_accuracy(data_loader)
        self.metrics.clean_accuracy = clean_acc
        self.metrics.clean_confidence = clean_conf
        logger.info(f"Clean Accuracy: {clean_acc:.2%} (Confidence: {clean_conf:.4f})")
        
        # Test each attack
        robust_accuracies = []
        for attack_type in attack_types:
            logger.info(f"\nTesting {attack_type.upper()} attack...")
            
            try:
                results = self.test_adversarial_attack(
                    data_loader, attack_type, use_defense
                )
                
                # Store metrics
                self.metrics.adversarial_accuracy[attack_type] = results['accuracy']
                self.metrics.adversarial_confidence[attack_type] = results['confidence']
                self.metrics.l2_distance[attack_type] = results['l2_distance']
                self.metrics.linf_distance[attack_type] = results['linf_distance']
                self.metrics.attack_time[attack_type] = results['time']
                
                # Calculate success rate
                success_rate = 1.0 - (results['accuracy'] / clean_acc) if clean_acc > 0 else 0.0
                self.metrics.attack_success_rate[attack_type] = success_rate
                
                robust_accuracies.append(results['accuracy'])
                
                logger.info(
                    f"{attack_type.upper()} Results:\n"
                    f"  Accuracy: {results['accuracy']:.2%}\n"
                    f"  Success Rate: {success_rate:.2%}\n"
                    f"  L2 Distance: {results['l2_distance']:.4f}\n"
                    f"  L∞ Distance: {results['linf_distance']:.4f}\n"
                    f"  Time: {results['time']:.2f}s"
                )
                
            except Exception as e:
                logger.error(f"Error testing {attack_type}: {e}")
                self.metrics.adversarial_accuracy[attack_type] = 0.0
                self.metrics.attack_success_rate[attack_type] = 1.0
        
        # Calculate robust accuracy (minimum across all attacks)
        if robust_accuracies:
            self.metrics.robust_accuracy = min(robust_accuracies)
            self.metrics.successful_attacks = int(
                total_samples * (1 - self.metrics.robust_accuracy / clean_acc)
            ) if clean_acc > 0 else 0
        
        logger.info("\n" + self.metrics.summary())
        
        return self.metrics
    
    def test_defense_effectiveness(
        self,
        data_loader: DataLoader,
        attack_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare performance with and without defenses
        
        Args:
            data_loader: Test data loader
            attack_types: List of attacks to test
            
        Returns:
            Comparison results
        """
        logger.info("Testing defense effectiveness...")
        
        # Test without defense
        logger.info("\n=== Without Defense ===")
        metrics_no_defense = self.test_comprehensive(
            data_loader, attack_types, use_defense=False
        )
        
        # Test with defense
        logger.info("\n=== With Defense ===")
        metrics_with_defense = self.test_comprehensive(
            data_loader, attack_types, use_defense=True
        )
        
        # Calculate improvements
        improvements = {}
        for attack_type in metrics_no_defense.adversarial_accuracy:
            acc_no_def = metrics_no_defense.adversarial_accuracy[attack_type]
            acc_with_def = metrics_with_defense.adversarial_accuracy[attack_type]
            improvement = acc_with_def - acc_no_def
            improvements[attack_type] = {
                'absolute_improvement': improvement,
                'relative_improvement': improvement / acc_no_def if acc_no_def > 0 else 0.0
            }
        
        results = {
            'without_defense': metrics_no_defense.to_dict(),
            'with_defense': metrics_with_defense.to_dict(),
            'improvements': improvements
        }
        
        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("DEFENSE EFFECTIVENESS SUMMARY")
        logger.info("=" * 60)
        for attack_type, improvement in improvements.items():
            logger.info(
                f"{attack_type.upper()}:\n"
                f"  Absolute Improvement: {improvement['absolute_improvement']:.2%}\n"
                f"  Relative Improvement: {improvement['relative_improvement']:.2%}"
            )
        logger.info("=" * 60)
        
        return results
    
    def save_results(self, filepath: str):
        """Save test results to file"""
        results = self.metrics.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def generate_report(self, filepath: str):
        """Generate detailed test report"""
        report_lines = [
            "# Adversarial Robustness Test Report",
            "",
            f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Device**: {self.device}",
            f"**Total Samples**: {self.metrics.total_samples}",
            "",
            "## Summary Statistics",
            "",
            f"- **Clean Accuracy**: {self.metrics.clean_accuracy:.2%}",
            f"- **Robust Accuracy**: {self.metrics.robust_accuracy:.2%}",
            f"- **Successful Attacks**: {self.metrics.successful_attacks}",
            f"- **Detected Attacks**: {self.metrics.detected_attacks}",
            "",
            "## Attack Results",
            ""
        ]
        
        # Add table
        report_lines.extend([
            "| Attack | Accuracy | Success Rate | L2 Dist | L∞ Dist | Time |",
            "|--------|----------|--------------|---------|---------|------|"
        ])
        
        for attack_type in self.metrics.adversarial_accuracy:
            acc = self.metrics.adversarial_accuracy[attack_type]
            sr = self.metrics.attack_success_rate[attack_type]
            l2 = self.metrics.l2_distance.get(attack_type, 0)
            linf = self.metrics.linf_distance.get(attack_type, 0)
            t = self.metrics.attack_time.get(attack_type, 0)
            
            report_lines.append(
                f"| {attack_type.upper()} | {acc:.2%} | {sr:.2%} | "
                f"{l2:.4f} | {linf:.4f} | {t:.2f}s |"
            )
        
        report_lines.extend([
            "",
            "## Recommendations",
            "",
            self._generate_recommendations(),
            ""
        ])
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Report saved to {filepath}")
    
    def _generate_recommendations(self) -> str:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if self.metrics.robust_accuracy < 0.5:
            recommendations.append(
                "⚠️ **Critical**: Robust accuracy is very low. "
                "Consider adversarial training or stronger defenses."
            )
        elif self.metrics.robust_accuracy < 0.7:
            recommendations.append(
                "⚠️ **Warning**: Robust accuracy needs improvement. "
                "Consider additional defense mechanisms."
            )
        
        # Check individual attacks
        for attack_type, success_rate in self.metrics.attack_success_rate.items():
            if success_rate > 0.8:
                recommendations.append(
                    f"⚠️ {attack_type.upper()} attack has high success rate. "
                    f"Model is vulnerable to this attack type."
                )
        
        if not recommendations:
            recommendations.append(
                "✅ Model shows good robustness against tested attacks."
            )
        
        return '\n'.join(f"- {r}" for r in recommendations)
