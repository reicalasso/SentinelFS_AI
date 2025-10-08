"""
A/B Testing Framework for Model Comparison

Provides A/B testing capabilities for comparing model versions in production
with comprehensive metrics tracking and statistical analysis.
"""

import json
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging


class TestStatus(Enum):
    """A/B test status."""
    SETUP = "setup"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class TestMetrics:
    """Metrics for a model in an A/B test."""
    model_version: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    avg_confidence: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def precision(self) -> float:
        """Calculate precision."""
        tp_fp = self.true_positives + self.false_positives
        if tp_fp == 0:
            return 0.0
        return self.true_positives / tp_fp
    
    @property
    def recall(self) -> float:
        """Calculate recall."""
        tp_fn = self.true_positives + self.false_negatives
        if tp_fn == 0:
            return 0.0
        return self.true_positives / tp_fn
    
    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['computed_metrics'] = {
            'error_rate': self.error_rate,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestMetrics':
        """Create from dictionary."""
        # Remove computed metrics if present
        data = {k: v for k, v in data.items() if k != 'computed_metrics'}
        return cls(**data)


@dataclass
class ABTest:
    """A/B test configuration and state."""
    test_id: str
    test_name: str
    model_a: str  # Control/baseline model
    model_b: str  # Challenger model
    traffic_split: float = 0.5  # % of traffic to model_b (0.0-1.0)
    status: str = TestStatus.SETUP.value
    created_at: str = ""
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    created_by: str = "system"
    description: str = ""
    min_samples: int = 100  # Minimum samples before declaring winner
    significance_threshold: float = 0.05  # P-value threshold
    metrics_a: Optional[TestMetrics] = None
    metrics_b: Optional[TestMetrics] = None
    winner: Optional[str] = None
    winner_confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.metrics_a:
            data['metrics_a'] = self.metrics_a.to_dict()
        if self.metrics_b:
            data['metrics_b'] = self.metrics_b.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ABTest':
        """Create from dictionary."""
        if 'metrics_a' in data and data['metrics_a']:
            data['metrics_a'] = TestMetrics.from_dict(data['metrics_a'])
        if 'metrics_b' in data and data['metrics_b']:
            data['metrics_b'] = TestMetrics.from_dict(data['metrics_b'])
        return cls(**data)


class ABTestManager:
    """
    Manages A/B tests for model comparison.
    
    Features:
    - A/B test creation and configuration
    - Traffic splitting and routing
    - Metrics collection and analysis
    - Statistical significance testing
    - Automated winner determination
    """
    
    def __init__(self, tests_dir: str = "models/ab_tests"):
        """
        Initialize A/B test manager.
        
        Args:
            tests_dir: Directory for A/B test data
        """
        self.tests_dir = Path(tests_dir)
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Active tests
        self.tests: Dict[str, ABTest] = {}
        
        # Load existing tests
        self._load_tests()
    
    def create_test(
        self,
        test_name: str,
        model_a: str,
        model_b: str,
        traffic_split: float = 0.5,
        description: str = "",
        created_by: str = "system",
        min_samples: int = 100,
        significance_threshold: float = 0.05
    ) -> ABTest:
        """
        Create a new A/B test.
        
        Args:
            test_name: Test name
            model_a: Control/baseline model version
            model_b: Challenger model version
            traffic_split: % of traffic to model_b (0.0-1.0)
            description: Test description
            created_by: User creating the test
            min_samples: Minimum samples for significance
            significance_threshold: P-value threshold
        
        Returns:
            Created ABTest
        """
        # Validate traffic split
        if not 0.0 <= traffic_split <= 1.0:
            raise ValueError("traffic_split must be between 0.0 and 1.0")
        
        # Generate test ID
        test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create test
        test = ABTest(
            test_id=test_id,
            test_name=test_name,
            model_a=model_a,
            model_b=model_b,
            traffic_split=traffic_split,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            description=description,
            min_samples=min_samples,
            significance_threshold=significance_threshold,
            metrics_a=TestMetrics(model_version=model_a),
            metrics_b=TestMetrics(model_version=model_b)
        )
        
        self.tests[test_id] = test
        self._save_test(test)
        
        self.logger.info(
            f"Created A/B test {test_id}: {model_a} vs {model_b}"
        )
        
        return test
    
    def start_test(self, test_id: str):
        """Start an A/B test."""
        test = self.tests.get(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")
        
        if test.status != TestStatus.SETUP.value:
            raise ValueError(f"Test must be in SETUP status to start")
        
        test.status = TestStatus.RUNNING.value
        test.started_at = datetime.now().isoformat()
        
        self._save_test(test)
        self.logger.info(f"Started A/B test {test_id}")
    
    def pause_test(self, test_id: str):
        """Pause a running A/B test."""
        test = self.tests.get(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")
        
        if test.status != TestStatus.RUNNING.value:
            raise ValueError(f"Test must be RUNNING to pause")
        
        test.status = TestStatus.PAUSED.value
        self._save_test(test)
        
        self.logger.info(f"Paused A/B test {test_id}")
    
    def resume_test(self, test_id: str):
        """Resume a paused A/B test."""
        test = self.tests.get(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")
        
        if test.status != TestStatus.PAUSED.value:
            raise ValueError(f"Test must be PAUSED to resume")
        
        test.status = TestStatus.RUNNING.value
        self._save_test(test)
        
        self.logger.info(f"Resumed A/B test {test_id}")
    
    def route_request(self, test_id: str) -> str:
        """
        Route a request to model A or B based on traffic split.
        
        Args:
            test_id: Test ID
        
        Returns:
            Selected model version
        """
        test = self.tests.get(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")
        
        if test.status != TestStatus.RUNNING.value:
            # Default to model A if test not running
            return test.model_a
        
        # Route based on traffic split
        if random.random() < test.traffic_split:
            return test.model_b
        else:
            return test.model_a
    
    def record_result(
        self,
        test_id: str,
        model_version: str,
        success: bool,
        latency_ms: float,
        confidence: Optional[float] = None,
        true_label: Optional[bool] = None,
        predicted_label: Optional[bool] = None,
        custom_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Record a result from model inference.
        
        Args:
            test_id: Test ID
            model_version: Model version that handled the request
            success: Whether request was successful
            latency_ms: Request latency in milliseconds
            confidence: Model confidence score
            true_label: True label (if available)
            predicted_label: Predicted label
            custom_metrics: Additional custom metrics
        """
        test = self.tests.get(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")
        
        # Determine which metrics to update
        if model_version == test.model_a:
            metrics = test.metrics_a
        elif model_version == test.model_b:
            metrics = test.metrics_b
        else:
            raise ValueError(f"Unknown model version: {model_version}")
        
        # Update metrics
        metrics.total_requests += 1
        
        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
        
        # Update rolling average latency
        n = metrics.total_requests
        metrics.avg_latency_ms = (
            (metrics.avg_latency_ms * (n - 1) + latency_ms) / n
        )
        
        # Update confidence
        if confidence is not None:
            metrics.avg_confidence = (
                (metrics.avg_confidence * (n - 1) + confidence) / n
            )
        
        # Update confusion matrix if labels provided
        if true_label is not None and predicted_label is not None:
            if true_label and predicted_label:
                metrics.true_positives += 1
            elif not true_label and predicted_label:
                metrics.false_positives += 1
            elif true_label and not predicted_label:
                metrics.false_negatives += 1
            else:  # not true_label and not predicted_label
                metrics.true_negatives += 1
        
        # Update custom metrics
        if custom_metrics:
            for key, value in custom_metrics.items():
                if key in metrics.custom_metrics:
                    # Rolling average
                    metrics.custom_metrics[key] = (
                        (metrics.custom_metrics[key] * (n - 1) + value) / n
                    )
                else:
                    metrics.custom_metrics[key] = value
        
        self._save_test(test)
    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """
        Get current test results and analysis.
        
        Args:
            test_id: Test ID
        
        Returns:
            Test results and analysis
        """
        test = self.tests.get(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")
        
        results = {
            "test_id": test_id,
            "test_name": test.test_name,
            "status": test.status,
            "model_a": test.model_a,
            "model_b": test.model_b,
            "metrics_a": test.metrics_a.to_dict() if test.metrics_a else {},
            "metrics_b": test.metrics_b.to_dict() if test.metrics_b else {},
            "comparison": self._compare_metrics(test.metrics_a, test.metrics_b),
            "statistical_significance": self._calculate_significance(test),
            "recommendation": self._get_recommendation(test)
        }
        
        return results
    
    def complete_test(self, test_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Complete an A/B test and determine winner.
        
        Args:
            test_id: Test ID
            force: Force completion even without minimum samples
        
        Returns:
            Final test results
        """
        test = self.tests.get(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")
        
        if test.status not in [TestStatus.RUNNING.value, TestStatus.PAUSED.value]:
            raise ValueError(f"Test must be RUNNING or PAUSED to complete")
        
        # Check minimum samples
        if not force:
            total_samples = test.metrics_a.total_requests + test.metrics_b.total_requests
            if total_samples < test.min_samples:
                raise ValueError(
                    f"Not enough samples: {total_samples} < {test.min_samples}. "
                    f"Use force=True to complete anyway."
                )
        
        # Determine winner
        winner_info = self._determine_winner(test)
        test.winner = winner_info['winner']
        test.winner_confidence = winner_info['confidence']
        
        # Update status
        test.status = TestStatus.COMPLETED.value
        test.ended_at = datetime.now().isoformat()
        
        self._save_test(test)
        
        self.logger.info(
            f"Completed A/B test {test_id}. Winner: {test.winner} "
            f"(confidence: {test.winner_confidence:.2%})"
        )
        
        return self.get_test_results(test_id)
    
    def _compare_metrics(
        self,
        metrics_a: TestMetrics,
        metrics_b: TestMetrics
    ) -> Dict[str, Any]:
        """Compare metrics between two models."""
        comparison = {}
        
        # Compare key metrics
        metrics_to_compare = [
            ('error_rate', 'lower_is_better'),
            ('avg_latency_ms', 'lower_is_better'),
            ('precision', 'higher_is_better'),
            ('recall', 'higher_is_better'),
            ('f1_score', 'higher_is_better'),
            ('avg_confidence', 'higher_is_better')
        ]
        
        for metric_name, direction in metrics_to_compare:
            if metric_name.startswith('avg_') or metric_name in ['precision', 'recall', 'f1_score', 'error_rate']:
                val_a = getattr(metrics_a, metric_name, 0.0) if hasattr(metrics_a, metric_name) else metrics_a.__dict__.get(metric_name, 0.0)
                val_b = getattr(metrics_b, metric_name, 0.0) if hasattr(metrics_b, metric_name) else metrics_b.__dict__.get(metric_name, 0.0)
                
                diff = val_b - val_a
                percent_change = (diff / val_a * 100) if val_a != 0 else 0.0
                
                if direction == 'lower_is_better':
                    winner = 'B' if val_b < val_a else 'A'
                else:
                    winner = 'B' if val_b > val_a else 'A'
                
                comparison[metric_name] = {
                    'model_a': val_a,
                    'model_b': val_b,
                    'difference': diff,
                    'percent_change': percent_change,
                    'winner': winner
                }
        
        return comparison
    
    def _calculate_significance(self, test: ABTest) -> Dict[str, Any]:
        """Calculate statistical significance (simplified)."""
        metrics_a = test.metrics_a
        metrics_b = test.metrics_b
        
        # Simple significance check based on sample size and difference
        total_samples = metrics_a.total_requests + metrics_b.total_requests
        
        if total_samples < test.min_samples:
            return {
                "is_significant": False,
                "reason": f"Insufficient samples: {total_samples} < {test.min_samples}",
                "confidence": 0.0
            }
        
        # Compare F1 scores (or error rates)
        f1_a = metrics_a.f1_score
        f1_b = metrics_b.f1_score
        
        # Simplified significance: difference > 1% with enough samples
        diff = abs(f1_b - f1_a)
        is_significant = diff > 0.01 and total_samples >= test.min_samples
        
        confidence = min(0.99, (total_samples / test.min_samples) * diff)
        
        return {
            "is_significant": is_significant,
            "f1_difference": diff,
            "confidence": confidence,
            "total_samples": total_samples
        }
    
    def _determine_winner(self, test: ABTest) -> Dict[str, Any]:
        """Determine the winner of an A/B test."""
        metrics_a = test.metrics_a
        metrics_b = test.metrics_b
        
        # Primary metric: F1 score
        f1_a = metrics_a.f1_score
        f1_b = metrics_b.f1_score
        
        # Secondary metrics
        latency_a = metrics_a.avg_latency_ms
        latency_b = metrics_b.avg_latency_ms
        
        # Scoring system
        score_a = 0
        score_b = 0
        
        # F1 score (most important)
        if f1_a > f1_b:
            score_a += 3
        elif f1_b > f1_a:
            score_b += 3
        
        # Latency (important)
        if latency_a < latency_b:
            score_a += 2
        elif latency_b < latency_a:
            score_b += 2
        
        # Error rate (important)
        if metrics_a.error_rate < metrics_b.error_rate:
            score_a += 2
        elif metrics_b.error_rate < metrics_a.error_rate:
            score_b += 2
        
        # Determine winner
        if score_a > score_b:
            winner = test.model_a
            confidence = score_a / (score_a + score_b)
        elif score_b > score_a:
            winner = test.model_b
            confidence = score_b / (score_a + score_b)
        else:
            winner = test.model_a  # Tie goes to baseline
            confidence = 0.5
        
        return {
            "winner": winner,
            "confidence": confidence,
            "score_a": score_a,
            "score_b": score_b
        }
    
    def _get_recommendation(self, test: ABTest) -> str:
        """Get recommendation based on test results."""
        metrics_a = test.metrics_a
        metrics_b = test.metrics_b
        
        total_samples = metrics_a.total_requests + metrics_b.total_requests
        
        if total_samples < test.min_samples:
            return f"Continue test. Need {test.min_samples - total_samples} more samples."
        
        winner_info = self._determine_winner(test)
        winner = winner_info['winner']
        confidence = winner_info['confidence']
        
        if confidence > 0.8:
            if winner == test.model_b:
                return f"Strong recommendation to promote Model B ({winner}). Confidence: {confidence:.2%}"
            else:
                return f"Keep Model A ({winner}) in production. Model B shows no improvement."
        elif confidence > 0.6:
            return f"Moderate recommendation for {winner}. Consider extended testing."
        else:
            return "Results inconclusive. Extend test or adjust traffic split."
    
    def _save_test(self, test: ABTest):
        """Save test to file."""
        test_path = self.tests_dir / f"{test.test_id}.json"
        with open(test_path, 'w') as f:
            json.dump(test.to_dict(), f, indent=2)
    
    def _load_tests(self):
        """Load all existing tests."""
        for test_path in self.tests_dir.glob("test_*.json"):
            try:
                with open(test_path, 'r') as f:
                    test_data = json.load(f)
                
                test = ABTest.from_dict(test_data)
                self.tests[test.test_id] = test
            
            except Exception as e:
                self.logger.error(f"Error loading test from {test_path}: {e}")
