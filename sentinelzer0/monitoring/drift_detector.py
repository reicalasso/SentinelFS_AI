"""
Model Drift Detection System

This module implements statistical methods to detect when the model's
performance degrades or when the input data distribution changes.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from typing import List, Dict, Optional, Tuple
import logging
import time
from dataclasses import dataclass
from collections import deque

from .metrics import update_drift_score, record_alert

logger = logging.getLogger(__name__)

@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    has_drift: bool
    drift_score: float
    confidence: float
    method: str
    details: Dict[str, float]

class ModelDriftDetector:
    """
    Detects model drift using multiple statistical methods.

    Methods implemented:
    - Kolmogorov-Smirnov test for distribution changes
    - Population Stability Index (PSI)
    - Jensen-Shannon divergence
    - Prediction confidence monitoring
    """

    def __init__(
        self,
        window_size: int = 1000,
        drift_threshold: float = 0.05,
        alert_threshold: float = 0.1,
        methods: List[str] = None
    ):
        """
        Initialize the drift detector.

        Args:
            window_size: Number of recent predictions to keep for comparison
            drift_threshold: Threshold for detecting drift (0.0-1.0)
            alert_threshold: Threshold for triggering alerts (0.0-1.0)
            methods: List of drift detection methods to use
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.alert_threshold = alert_threshold

        # Available methods
        self.available_methods = [
            'ks_test',      # Kolmogorov-Smirnov test
            'psi',          # Population Stability Index
            'js_divergence', # Jensen-Shannon divergence
            'confidence'    # Prediction confidence monitoring
        ]

        self.methods = methods or self.available_methods

        # Historical data storage
        self.prediction_scores = deque(maxlen=window_size)
        self.reference_scores = []  # Baseline scores for comparison
        self.baseline_set = False

        # Drift tracking
        self.drift_history = []
        self.last_drift_check = 0
        self.check_interval = 300  # Check every 5 minutes

        logger.info(f"Initialized drift detector with methods: {self.methods}")

    def set_baseline(self, scores: List[float]):
        """
        Set the baseline distribution for drift detection.

        Args:
            scores: List of prediction scores from normal operation
        """
        self.reference_scores = np.array(scores)
        self.baseline_set = True
        logger.info(f"Set baseline with {len(scores)} reference scores")

    def add_prediction(self, score: float):
        """
        Add a new prediction score to the monitoring window.

        Args:
            score: Prediction score (0.0-1.0)
        """
        self.prediction_scores.append(score)

        # Periodic drift check
        current_time = time.time()
        if (current_time - self.last_drift_check) > self.check_interval:
            self._check_drift()
            self.last_drift_check = current_time

    def _check_drift(self):
        """Perform drift detection if we have enough data."""
        if not self.baseline_set or len(self.prediction_scores) < 100:
            return

        results = []
        for method in self.methods:
            try:
                result = getattr(self, f'_detect_drift_{method}')()
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Error in drift detection method {method}: {e}")

        if results:
            # Combine results (take maximum drift score)
            max_drift = max(results, key=lambda x: x.drift_score)
            self._handle_drift_result(max_drift)

    def _detect_drift_ks_test(self) -> Optional[DriftDetectionResult]:
        """Detect drift using Kolmogorov-Smirnov test."""
        if len(self.prediction_scores) < 50:
            return None

        current_scores = np.array(list(self.prediction_scores))

        try:
            statistic, p_value = stats.ks_2samp(
                self.reference_scores,
                current_scores
            )

            # Convert to drift score (0-1 scale)
            drift_score = min(1.0, statistic * 2)  # KS statistic ranges 0-1, scale it

            return DriftDetectionResult(
                has_drift=drift_score > self.drift_threshold,
                drift_score=drift_score,
                confidence=1 - p_value,  # Higher confidence = lower p-value
                method='ks_test',
                details={
                    'ks_statistic': statistic,
                    'p_value': p_value,
                    'sample_size': len(current_scores)
                }
            )
        except Exception as e:
            logger.warning(f"KS test failed: {e}")
            return None

    def _detect_drift_psi(self) -> Optional[DriftDetectionResult]:
        """Detect drift using Population Stability Index (PSI)."""
        if len(self.prediction_scores) < 50:
            return None

        current_scores = np.array(list(self.prediction_scores))

        try:
            # Create histograms
            bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1

            ref_hist, _ = np.histogram(self.reference_scores, bins=bins, density=True)
            curr_hist, _ = np.histogram(current_scores, bins=bins, density=True)

            # Avoid division by zero
            ref_hist = np.where(ref_hist == 0, 1e-6, ref_hist)
            curr_hist = np.where(curr_hist == 0, 1e-6, curr_hist)

            # Calculate PSI
            psi = np.sum((curr_hist - ref_hist) * np.log(curr_hist / ref_hist))

            # PSI > 0.1 indicates significant change
            # PSI > 0.25 indicates major change
            drift_score = min(1.0, psi / 0.5)  # Normalize to 0-1

            return DriftDetectionResult(
                has_drift=psi > 0.1,  # Standard PSI threshold
                drift_score=drift_score,
                confidence=min(1.0, psi / 0.25),  # Confidence based on PSI magnitude
                method='psi',
                details={
                    'psi_score': psi,
                    'sample_size': len(current_scores)
                }
            )
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return None

    def _detect_drift_js_divergence(self) -> Optional[DriftDetectionResult]:
        """Detect drift using Jensen-Shannon divergence."""
        if len(self.prediction_scores) < 50:
            return None

        current_scores = np.array(list(self.prediction_scores))

        try:
            # Create probability distributions
            bins = np.linspace(0, 1, 21)  # 20 bins for better resolution

            ref_hist, _ = np.histogram(self.reference_scores, bins=bins, density=True)
            curr_hist, _ = np.histogram(current_scores, bins=bins, density=True)

            # Normalize to probability distributions
            ref_prob = ref_hist / ref_hist.sum()
            curr_prob = curr_hist / curr_hist.sum()

            # Calculate Jensen-Shannon divergence
            m = 0.5 * (ref_prob + curr_prob)
            js_div = 0.5 * (stats.entropy(ref_prob, m) + stats.entropy(curr_prob, m))

            # JS divergence ranges from 0 to 1
            drift_score = js_div

            return DriftDetectionResult(
                has_drift=js_div > self.drift_threshold,
                drift_score=drift_score,
                confidence=js_div,  # JS divergence itself indicates confidence
                method='js_divergence',
                details={
                    'js_divergence': js_div,
                    'sample_size': len(current_scores)
                }
            )
        except Exception as e:
            logger.warning(f"JS divergence calculation failed: {e}")
            return None

    def _detect_drift_confidence(self) -> Optional[DriftDetectionResult]:
        """Detect drift based on prediction confidence changes."""
        if len(self.prediction_scores) < 100:
            return None

        current_scores = np.array(list(self.prediction_scores))

        try:
            # Calculate confidence metrics
            current_mean = np.mean(current_scores)
            current_std = np.std(current_scores)
            reference_mean = np.mean(self.reference_scores)
            reference_std = np.std(self.reference_scores)

            # Check for significant changes in distribution parameters
            mean_diff = abs(current_mean - reference_mean)
            std_diff = abs(current_std - reference_std)

            # Normalize differences
            mean_drift = min(1.0, mean_diff / 0.2)  # 0.2 is significant mean change
            std_drift = min(1.0, std_diff / 0.1)   # 0.1 is significant std change

            drift_score = max(mean_drift, std_drift)

            return DriftDetectionResult(
                has_drift=drift_score > self.drift_threshold,
                drift_score=drift_score,
                confidence=drift_score,  # Score indicates confidence
                method='confidence',
                details={
                    'current_mean': current_mean,
                    'reference_mean': reference_mean,
                    'current_std': current_std,
                    'reference_std': reference_std,
                    'mean_diff': mean_diff,
                    'std_diff': std_diff
                }
            )
        except Exception as e:
            logger.warning(f"Confidence monitoring failed: {e}")
            return None

    def _handle_drift_result(self, result: DriftDetectionResult):
        """Handle drift detection result."""
        # Update metrics
        update_drift_score(result.drift_score)

        # Store in history
        self.drift_history.append({
            'timestamp': time.time(),
            'method': result.method,
            'drift_score': result.drift_score,
            'has_drift': result.has_drift,
            'confidence': result.confidence
        })

        # Keep only recent history
        if len(self.drift_history) > 100:
            self.drift_history = self.drift_history[-100:]

        # Alert if threshold exceeded
        if result.drift_score > self.alert_threshold:
            record_alert('drift', 'warning')
            logger.warning(
                f"Model drift detected! Method: {result.method}, "
                f"Score: {result.drift_score:.3f}, Confidence: {result.confidence:.3f}"
            )

            # Log detailed information
            for key, value in result.details.items():
                logger.info(f"Drift detail - {key}: {value}")

    def get_drift_status(self) -> Dict:
        """
        Get current drift detection status.

        Returns:
            Dictionary with drift status information
        """
        if not self.drift_history:
            return {
                'has_drift': False,
                'drift_score': 0.0,
                'last_check': self.last_drift_check,
                'baseline_set': self.baseline_set,
                'samples_collected': len(self.prediction_scores)
            }

        latest = self.drift_history[-1]
        return {
            'has_drift': latest['has_drift'],
            'drift_score': latest['drift_score'],
            'method': latest['method'],
            'confidence': latest['confidence'],
            'last_check': latest['timestamp'],
            'baseline_set': self.baseline_set,
            'samples_collected': len(self.prediction_scores),
            'history_size': len(self.drift_history)
        }

    def reset_baseline(self):
        """Reset the baseline with current data."""
        if len(self.prediction_scores) >= 100:
            self.set_baseline(list(self.prediction_scores))
            logger.info("Reset baseline with current prediction distribution")
        else:
            logger.warning("Not enough data to reset baseline")