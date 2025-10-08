"""
Security Engine Base Classes and Core Logic

This module provides the foundation for the security engine including
base classes, detection methods, and integration interfaces.
"""

import abc
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DetectionMethod(Enum):
    """Enumeration of available detection methods."""
    AI_MODEL = "ai_model"
    YARA_SIGNATURES = "yara_signatures"
    ENTROPY_ANALYSIS = "entropy_analysis"
    HEURISTIC_RULES = "heuristic_rules"
    CONTENT_INSPECTION = "content_inspection"


class ThreatLevel(Enum):
    """Threat level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityResult:
    """Result from a security detection method."""
    method: DetectionMethod
    score: float  # 0.0 to 1.0
    threat_level: ThreatLevel
    confidence: float  # 0.0 to 1.0
    details: Dict[str, Any]
    matched_rules: List[str] = None

    def __post_init__(self):
        if self.matched_rules is None:
            self.matched_rules = []


@dataclass
class CombinedResult:
    """Combined result from multiple detection methods."""
    ai_score: float
    security_score: float
    combined_score: float
    threat_level: ThreatLevel
    detection_methods: List[DetectionMethod]
    results: List[SecurityResult]
    correlation_factors: Dict[str, float]
    final_decision: bool


class BaseDetector(abc.ABC):
    """Abstract base class for all security detectors."""

    def __init__(self, name: str, method: DetectionMethod):
        self.name = name
        self.method = method
        self.logger = get_logger(f"{__name__}.{name}")

    @abc.abstractmethod
    def analyze(self, data: Any, context: Optional[Dict[str, Any]] = None) -> SecurityResult:
        """Analyze data and return security result."""
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if this detector is available and properly configured."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            'name': self.name,
            'method': self.method.value,
            'available': self.is_available()
        }


class SecurityEngine:
    """
    Main security engine that coordinates multiple detection methods.

    This engine combines AI-based detection with traditional security methods
    to provide comprehensive threat detection.
    """

    def __init__(self):
        self.detectors: Dict[DetectionMethod, BaseDetector] = {}
        self.correlator = None
        self.logger = get_logger(__name__)

        # Initialize built-in detectors
        self._initialize_detectors()

    def _initialize_detectors(self):
        """Initialize all available detectors."""
        try:
            from .yara_detector import YaraDetector
            self.register_detector(YaraDetector())
        except ImportError:
            self.logger.warning("YARA detector not available")

        try:
            from .entropy_analyzer import EntropyAnalyzer
            self.register_detector(EntropyAnalyzer())
        except ImportError:
            self.logger.warning("Entropy analyzer not available")

        try:
            from .content_inspector import ContentInspector
            self.register_detector(ContentInspector())
        except ImportError:
            self.logger.warning("Content inspector not available")

        try:
            from .threat_correlator import ThreatCorrelator
            self.correlator = ThreatCorrelator()
        except ImportError:
            self.logger.warning("Threat correlator not available")

    def register_detector(self, detector: BaseDetector):
        """Register a new detector."""
        if detector.is_available():
            self.detectors[detector.method] = detector
            self.logger.info(f"Registered detector: {detector.name}")
        else:
            self.logger.warning(f"Detector {detector.name} is not available")

    def analyze_file(self, file_path: str, file_data: Optional[bytes] = None,
                    ai_score: Optional[float] = None) -> CombinedResult:
        """
        Analyze a file using all available detection methods.

        Args:
            file_path: Path to the file
            file_data: Optional file content as bytes
            ai_score: Optional AI model score (0.0 to 1.0)

        Returns:
            CombinedResult with analysis from all methods
        """
        results = []
        context = {
            'file_path': file_path,
            'file_data': file_data,
            'ai_score': ai_score
        }

        # Run all detectors
        for method, detector in self.detectors.items():
            try:
                result = detector.analyze(file_data or file_path, context)
                results.append(result)
                self.logger.debug(f"{method.value}: score={result.score:.3f}, "
                                f"threat={result.threat_level.value}")
            except Exception as e:
                self.logger.error(f"Error in {method.value} detector: {e}")

        # Combine results
        return self._combine_results(results, ai_score)

    def _combine_results(self, results: List[SecurityResult],
                        ai_score: Optional[float]) -> CombinedResult:
        """Combine results from multiple detection methods."""

        # Extract scores
        ai_score = ai_score or 0.0
        security_scores = [r.score for r in results]
        security_score = np.mean(security_scores) if security_scores else 0.0

        # Calculate combined score with weights
        weights = {
            DetectionMethod.AI_MODEL: 0.4,
            DetectionMethod.YARA_SIGNATURES: 0.3,
            DetectionMethod.ENTROPY_ANALYSIS: 0.2,
            DetectionMethod.CONTENT_INSPECTION: 0.1
        }

        combined_score = ai_score * weights[DetectionMethod.AI_MODEL]
        for result in results:
            weight = weights.get(result.method, 0.1)
            combined_score += result.score * weight

        # Determine threat level
        if combined_score >= 0.8:
            threat_level = ThreatLevel.CRITICAL
        elif combined_score >= 0.6:
            threat_level = ThreatLevel.HIGH
        elif combined_score >= 0.4:
            threat_level = ThreatLevel.MEDIUM
        else:
            threat_level = ThreatLevel.LOW

        # Apply correlation if available
        correlation_factors = {}
        if self.correlator:
            correlation_factors = self.correlator.correlate(results)

        # Final decision (can be overridden by correlation)
        final_decision = combined_score >= 0.5

        return CombinedResult(
            ai_score=ai_score,
            security_score=security_score,
            combined_score=combined_score,
            threat_level=threat_level,
            detection_methods=[r.method for r in results],
            results=results,
            correlation_factors=correlation_factors,
            final_decision=final_decision
        )

    def get_status(self) -> Dict[str, Any]:
        """Get status of all detectors."""
        return {
            'detectors': {method.value: detector.get_config()
                         for method, detector in self.detectors.items()},
            'correlator_available': self.correlator is not None,
            'total_detectors': len(self.detectors)
        }