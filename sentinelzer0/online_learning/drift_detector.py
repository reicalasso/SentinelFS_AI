"""
Concept Drift Detection

Detects distribution shifts and concept drift in data streams using
statistical tests and change detection algorithms.
"""

import numpy as np
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import logging
from scipy import stats


class DriftDetectionMethod(Enum):
    """Drift detection method options."""
    ADWIN = "adwin"  # Adaptive Windowing
    DDM = "ddm"  # Drift Detection Method
    KSWIN = "kswin"  # Kolmogorov-Smirnov Windowing
    PAGE_HINKLEY = "page_hinkley"  # Page-Hinkley Test
    STATISTICAL = "statistical"  # Simple statistical test


class ConceptDriftDetector:
    """
    Detect concept drift in data streams.
    
    Features:
    - Multiple drift detection algorithms
    - Statistical change detection
    - Window-based monitoring
    - False alarm control
    - Drift severity estimation
    """
    
    def __init__(
        self,
        method: DriftDetectionMethod = DriftDetectionMethod.ADWIN,
        window_size: int = 100,
        min_samples: int = 30,
        warning_threshold: float = 2.0,
        drift_threshold: float = 3.0,
        significance_level: float = 0.05
    ):
        """
        Initialize drift detector.
        
        Args:
            method: Drift detection method
            window_size: Size of sliding window
            min_samples: Minimum samples before detection
            warning_threshold: Threshold for warning
            drift_threshold: Threshold for drift detection
            significance_level: Statistical significance level
        """
        self.logger = logging.getLogger(__name__)
        self.method = method
        self.window_size = window_size
        self.min_samples = min_samples
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.significance_level = significance_level
        
        # Data windows
        self.reference_window = deque(maxlen=window_size)
        self.current_window = deque(maxlen=window_size)
        
        # DDM specific
        self.p_min = float('inf')
        self.s_min = float('inf')
        
        # Page-Hinkley specific
        self.ph_sum = 0.0
        self.ph_min = float('inf')
        
        # Statistics
        self.drift_count = 0
        self.warning_count = 0
        self.sample_count = 0
        self.last_drift_at = 0
        
        self.logger.info(f"Initialized drift detector with {method.value} method")
    
    def add_sample(self, value: float, is_correct: bool = True) -> Dict[str, Any]:
        """
        Add new sample and check for drift.
        
        Args:
            value: Sample value (e.g., prediction error, loss)
            is_correct: Whether prediction was correct
        
        Returns:
            Detection results
        """
        self.sample_count += 1
        
        # Add to current window
        self.current_window.append(value)
        
        # Check for drift based on method
        if self.method == DriftDetectionMethod.ADWIN:
            return self._detect_adwin(value)
        elif self.method == DriftDetectionMethod.DDM:
            return self._detect_ddm(is_correct)
        elif self.method == DriftDetectionMethod.KSWIN:
            return self._detect_kswin(value)
        elif self.method == DriftDetectionMethod.PAGE_HINKLEY:
            return self._detect_page_hinkley(value)
        elif self.method == DriftDetectionMethod.STATISTICAL:
            return self._detect_statistical(value)
        else:
            raise ValueError(f"Unknown drift detection method: {self.method}")
    
    def _detect_adwin(self, value: float) -> Dict[str, Any]:
        """
        ADWIN (Adaptive Windowing) drift detection.
        
        Maintains a window that automatically grows when data is stationary
        and shrinks when drift is detected.
        """
        if len(self.current_window) < self.min_samples:
            return {
                'drift_detected': False,
                'warning': False,
                'severity': 0.0,
                'samples': self.sample_count
            }
        
        # Simple ADWIN approximation: compare window halves
        window_list = list(self.current_window)
        split_point = len(window_list) // 2
        
        left_half = window_list[:split_point]
        right_half = window_list[split_point:]
        
        if len(left_half) < 10 or len(right_half) < 10:
            return {
                'drift_detected': False,
                'warning': False,
                'severity': 0.0,
                'samples': self.sample_count
            }
        
        # Statistical test (t-test)
        t_stat, p_value = stats.ttest_ind(left_half, right_half)
        
        # Detect drift
        drift_detected = p_value < self.significance_level
        warning = p_value < (self.significance_level * 2)
        
        severity = 1.0 - p_value if drift_detected else 0.0
        
        if drift_detected:
            self.drift_count += 1
            self.last_drift_at = self.sample_count
            self.reference_window = deque(right_half, maxlen=self.window_size)
            self.current_window.clear()
        elif warning:
            self.warning_count += 1
        
        return {
            'drift_detected': drift_detected,
            'warning': warning,
            'severity': severity,
            'p_value': p_value,
            'samples': self.sample_count
        }
    
    def _detect_ddm(self, is_correct: bool) -> Dict[str, Any]:
        """
        DDM (Drift Detection Method).
        
        Monitors error rate and its standard deviation.
        Drift is detected when error rate increases significantly.
        """
        # Update error rate
        error = 0 if is_correct else 1
        self.current_window.append(error)
        
        if len(self.current_window) < self.min_samples:
            return {
                'drift_detected': False,
                'warning': False,
                'severity': 0.0,
                'samples': self.sample_count
            }
        
        # Calculate error rate and standard deviation
        errors = list(self.current_window)
        p = np.mean(errors)  # Error rate
        s = np.std(errors)   # Standard deviation
        
        # Update minimum values
        if p + s < self.p_min + self.s_min:
            self.p_min = p
            self.s_min = s
        
        # Detect drift
        drift_threshold = self.p_min + self.drift_threshold * self.s_min
        warning_threshold = self.p_min + self.warning_threshold * self.s_min
        
        drift_detected = p + s > drift_threshold
        warning = p + s > warning_threshold
        
        severity = (p + s - self.p_min - self.s_min) / (self.s_min + 1e-10)
        
        if drift_detected:
            self.drift_count += 1
            self.last_drift_at = self.sample_count
            self.p_min = float('inf')
            self.s_min = float('inf')
            self.current_window.clear()
        elif warning:
            self.warning_count += 1
        
        return {
            'drift_detected': drift_detected,
            'warning': warning,
            'severity': max(0.0, severity),
            'error_rate': p,
            'samples': self.sample_count
        }
    
    def _detect_kswin(self, value: float) -> Dict[str, Any]:
        """
        KSWIN (Kolmogorov-Smirnov Windowing).
        
        Uses Kolmogorov-Smirnov test to compare distributions
        in sliding windows.
        """
        if len(self.reference_window) < self.min_samples:
            self.reference_window.append(value)
            return {
                'drift_detected': False,
                'warning': False,
                'severity': 0.0,
                'samples': self.sample_count
            }
        
        if len(self.current_window) < self.min_samples:
            return {
                'drift_detected': False,
                'warning': False,
                'severity': 0.0,
                'samples': self.sample_count
            }
        
        # Perform KS test
        ks_stat, p_value = stats.ks_2samp(
            list(self.reference_window),
            list(self.current_window)
        )
        
        # Detect drift
        drift_detected = p_value < self.significance_level
        warning = p_value < (self.significance_level * 2)
        
        severity = ks_stat if drift_detected else 0.0
        
        if drift_detected:
            self.drift_count += 1
            self.last_drift_at = self.sample_count
            # Update reference window
            self.reference_window = deque(
                list(self.current_window),
                maxlen=self.window_size
            )
            self.current_window.clear()
        elif warning:
            self.warning_count += 1
        
        return {
            'drift_detected': drift_detected,
            'warning': warning,
            'severity': severity,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'samples': self.sample_count
        }
    
    def _detect_page_hinkley(self, value: float) -> Dict[str, Any]:
        """
        Page-Hinkley Test.
        
        Cumulative sum-based change detection.
        """
        if len(self.current_window) < self.min_samples:
            return {
                'drift_detected': False,
                'warning': False,
                'severity': 0.0,
                'samples': self.sample_count
            }
        
        # Calculate mean
        mean_val = np.mean(list(self.current_window))
        
        # Update cumulative sum
        delta = 0.005  # Magnitude of change to detect
        self.ph_sum += value - mean_val - delta
        
        # Update minimum
        if self.ph_sum < self.ph_min:
            self.ph_min = self.ph_sum
        
        # Detect drift
        threshold = self.drift_threshold * np.std(list(self.current_window))
        warning_thresh = self.warning_threshold * np.std(list(self.current_window))
        
        diff = self.ph_sum - self.ph_min
        
        drift_detected = diff > threshold
        warning = diff > warning_thresh
        
        severity = diff / (threshold + 1e-10) if drift_detected else 0.0
        
        if drift_detected:
            self.drift_count += 1
            self.last_drift_at = self.sample_count
            self.ph_sum = 0.0
            self.ph_min = float('inf')
            self.current_window.clear()
        elif warning:
            self.warning_count += 1
        
        return {
            'drift_detected': drift_detected,
            'warning': warning,
            'severity': min(1.0, severity),
            'ph_value': diff,
            'samples': self.sample_count
        }
    
    def _detect_statistical(self, value: float) -> Dict[str, Any]:
        """
        Simple statistical drift detection.
        
        Compares current window statistics with reference window.
        """
        if len(self.reference_window) < self.min_samples:
            self.reference_window.append(value)
            return {
                'drift_detected': False,
                'warning': False,
                'severity': 0.0,
                'samples': self.sample_count
            }
        
        if len(self.current_window) < self.min_samples:
            return {
                'drift_detected': False,
                'warning': False,
                'severity': 0.0,
                'samples': self.sample_count
            }
        
        # Calculate means and standard deviations
        ref_mean = np.mean(list(self.reference_window))
        ref_std = np.std(list(self.reference_window))
        
        curr_mean = np.mean(list(self.current_window))
        curr_std = np.std(list(self.current_window))
        
        # Z-score for mean difference
        z_score = abs(curr_mean - ref_mean) / (ref_std + 1e-10)
        
        # Detect drift
        drift_detected = z_score > self.drift_threshold
        warning = z_score > self.warning_threshold
        
        severity = z_score / self.drift_threshold if drift_detected else 0.0
        
        if drift_detected:
            self.drift_count += 1
            self.last_drift_at = self.sample_count
            # Update reference
            self.reference_window = deque(
                list(self.current_window),
                maxlen=self.window_size
            )
            self.current_window.clear()
        elif warning:
            self.warning_count += 1
        
        return {
            'drift_detected': drift_detected,
            'warning': warning,
            'severity': min(1.0, severity),
            'z_score': z_score,
            'ref_mean': ref_mean,
            'curr_mean': curr_mean,
            'samples': self.sample_count
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get drift detection statistics."""
        return {
            'drift_count': self.drift_count,
            'warning_count': self.warning_count,
            'sample_count': self.sample_count,
            'last_drift_at': self.last_drift_at,
            'samples_since_drift': self.sample_count - self.last_drift_at,
            'reference_window_size': len(self.reference_window),
            'current_window_size': len(self.current_window),
            'method': self.method.value
        }
    
    def reset(self):
        """Reset drift detector."""
        self.reference_window.clear()
        self.current_window.clear()
        self.drift_count = 0
        self.warning_count = 0
        self.sample_count = 0
        self.last_drift_at = 0
        self.p_min = float('inf')
        self.s_min = float('inf')
        self.ph_sum = 0.0
        self.ph_min = float('inf')
        
        self.logger.info("Reset drift detector")
