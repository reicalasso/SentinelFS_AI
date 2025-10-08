"""
Threat Correlation Engine

This module correlates findings from multiple detection methods
to improve overall threat detection accuracy and reduce false positives.
"""

from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np

from .engine import BaseDetector, DetectionMethod, SecurityResult, ThreatLevel


class ThreatCorrelator:
    """
    Correlates threat intelligence from multiple detection sources.

    This class analyzes patterns across different detection methods to:
    - Reduce false positives through correlation
    - Increase confidence in threat detection
    - Identify complex attack patterns
    - Provide contextual threat intelligence
    """

    def __init__(self):
        self.correlation_rules = self._load_correlation_rules()

    def _load_correlation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined correlation rules."""
        return {
            "high_entropy_with_signatures": {
                "description": "High entropy + signature matches indicate packed malware",
                "methods": [DetectionMethod.ENTROPY_ANALYSIS, DetectionMethod.YARA_SIGNATURES],
                "correlation_factor": 1.5,
                "confidence_boost": 0.3
            },
            "ai_and_signatures": {
                "description": "AI + signature correlation increases confidence",
                "methods": [DetectionMethod.AI_MODEL, DetectionMethod.YARA_SIGNATURES],
                "correlation_factor": 1.3,
                "confidence_boost": 0.2
            },
            "entropy_and_content": {
                "description": "High entropy + suspicious content patterns",
                "methods": [DetectionMethod.ENTROPY_ANALYSIS, DetectionMethod.CONTENT_INSPECTION],
                "correlation_factor": 1.4,
                "confidence_boost": 0.25
            },
            "multi_method_consensus": {
                "description": "Multiple methods agreeing on threat",
                "methods": [],  # Any 3+ methods
                "min_methods": 3,
                "correlation_factor": 1.6,
                "confidence_boost": 0.4
            }
        }

    def correlate(self, results: List[SecurityResult]) -> Dict[str, float]:
        """
        Correlate results from multiple detection methods.

        Args:
            results: List of SecurityResult objects

        Returns:
            Dictionary of correlation factors and adjustments
        """
        if len(results) < 2:
            return {"correlation_factor": 1.0, "confidence_adjustment": 0.0}

        correlation_factors = {
            "correlation_factor": 1.0,
            "confidence_adjustment": 0.0,
            "rules_triggered": [],
            "method_agreement": 0.0
        }

        # Group results by method
        method_results = {r.method: r for r in results}

        # Apply correlation rules
        for rule_name, rule in self.correlation_rules.items():
            if self._rule_matches(rule, method_results):
                correlation_factors["correlation_factor"] *= rule["correlation_factor"]
                correlation_factors["confidence_adjustment"] += rule["confidence_boost"]
                correlation_factors["rules_triggered"].append(rule_name)

        # Calculate method agreement
        scores = [r.score for r in results]
        if scores:
            correlation_factors["method_agreement"] = np.std(scores)
            # Lower standard deviation = higher agreement
            agreement_factor = max(0, 1.0 - correlation_factors["method_agreement"])
            correlation_factors["confidence_adjustment"] += agreement_factor * 0.1

        # Cap correlation factor
        correlation_factors["correlation_factor"] = min(2.0, correlation_factors["correlation_factor"])
        correlation_factors["confidence_adjustment"] = min(0.5, correlation_factors["confidence_adjustment"])

        return correlation_factors

    def _rule_matches(self, rule: Dict[str, Any], method_results: Dict[DetectionMethod, SecurityResult]) -> bool:
        """Check if a correlation rule matches the current results."""
        required_methods = rule.get("methods", [])
        min_methods = rule.get("min_methods", 0)

        if required_methods:
            # Check if all required methods are present and have positive scores
            for method in required_methods:
                if method not in method_results:
                    return False
                if method_results[method].score < 0.3:  # Minimum threshold
                    return False
            return True

        elif min_methods > 0:
            # Check if minimum number of methods have positive results
            positive_results = [r for r in method_results.values() if r.score >= 0.3]
            return len(positive_results) >= min_methods

        return False

    def get_threat_context(self, results: List[SecurityResult]) -> Dict[str, Any]:
        """
        Generate contextual threat intelligence from correlated results.

        Args:
            results: List of SecurityResult objects

        Returns:
            Dictionary with threat context and recommendations
        """
        context = {
            "threat_pattern": "unknown",
            "confidence_level": "low",
            "recommended_actions": [],
            "risk_factors": [],
            "mitigation_priority": "low"
        }

        if not results:
            return context

        # Analyze threat patterns
        high_scoring_methods = [r for r in results if r.score >= 0.7]

        if len(high_scoring_methods) >= 2:
            context["threat_pattern"] = "multi_vector_attack"
            context["confidence_level"] = "high"
            context["recommended_actions"].extend([
                "Isolate affected system",
                "Collect full memory dump",
                "Check for lateral movement"
            ])
            context["mitigation_priority"] = "critical"

        elif any(r.method == DetectionMethod.YARA_SIGNATURES and r.score >= 0.8 for r in results):
            context["threat_pattern"] = "known_malware"
            context["confidence_level"] = "high"
            context["recommended_actions"].extend([
                "Quarantine file",
                "Update signature database",
                "Scan related systems"
            ])
            context["mitigation_priority"] = "high"

        elif any(r.method == DetectionMethod.ENTROPY_ANALYSIS and r.score >= 0.8 for r in results):
            context["threat_pattern"] = "suspicious_encryption"
            context["confidence_level"] = "medium"
            context["recommended_actions"].extend([
                "Monitor for data exfiltration",
                "Check file integrity",
                "Review access patterns"
            ])
            context["mitigation_priority"] = "medium"

        # Identify risk factors
        for result in results:
            if result.score >= 0.6:
                context["risk_factors"].append(f"{result.method.value}: {result.threat_level.value}")

        return context

    def calculate_combined_score(self, results: List[SecurityResult],
                               correlation_factors: Dict[str, float]) -> float:
        """
        Calculate final combined score with correlation adjustments.

        Args:
            results: List of SecurityResult objects
            correlation_factors: Correlation factors from correlate()

        Returns:
            Final combined threat score (0.0 to 1.0)
        """
        if not results:
            return 0.0

        # Base combined score (weighted average)
        weights = {
            DetectionMethod.AI_MODEL: 0.4,
            DetectionMethod.YARA_SIGNATURES: 0.3,
            DetectionMethod.ENTROPY_ANALYSIS: 0.2,
            DetectionMethod.CONTENT_INSPECTION: 0.1
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for result in results:
            weight = weights.get(result.method, 0.1)
            weighted_sum += result.score * weight
            total_weight += weight

        base_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Apply correlation factor
        correlation_factor = correlation_factors.get("correlation_factor", 1.0)
        final_score = base_score * correlation_factor

        # Apply confidence adjustment
        confidence_adjustment = correlation_factors.get("confidence_adjustment", 0.0)
        final_score += confidence_adjustment

        return min(1.0, max(0.0, final_score))