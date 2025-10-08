"""
Entropy Analysis for Encryption Detection

This module analyzes file entropy to detect encrypted or compressed content,
which may indicate malicious activity or data exfiltration.
"""

import math
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import os

from .engine import BaseDetector, DetectionMethod, SecurityResult, ThreatLevel


class EntropyAnalyzer(BaseDetector):
    """
    Entropy-based analysis for detecting encrypted/compressed files.

    High entropy often indicates encryption, compression, or packed executables.
    This detector analyzes byte distribution to identify suspicious entropy patterns.
    """

    def __init__(self, high_entropy_threshold: float = 7.5,
                 low_entropy_threshold: float = 2.0):
        super().__init__("Entropy Analyzer", DetectionMethod.ENTROPY_ANALYSIS)
        self.high_entropy_threshold = high_entropy_threshold
        self.low_entropy_threshold = low_entropy_threshold

    def analyze(self, data: Any, context: Optional[Dict[str, Any]] = None) -> SecurityResult:
        """
        Analyze entropy of file content.

        Args:
            data: File path (str) or file content (bytes)
            context: Additional context information

        Returns:
            SecurityResult with entropy analysis
        """
        try:
            # Get file content
            if isinstance(data, str) and os.path.isfile(data):
                with open(data, 'rb') as f:
                    content = f.read()
            elif isinstance(data, bytes):
                content = data
            else:
                return SecurityResult(
                    method=DetectionMethod.ENTROPY_ANALYSIS,
                    score=0.0,
                    threat_level=ThreatLevel.LOW,
                    confidence=0.0,
                    details={"error": "Invalid data type for entropy analysis"}
                )

            if len(content) == 0:
                return SecurityResult(
                    method=DetectionMethod.ENTROPY_ANALYSIS,
                    score=0.0,
                    threat_level=ThreatLevel.LOW,
                    confidence=1.0,
                    details={"entropy": 0.0, "file_size": 0, "reason": "Empty file"}
                )

            # Calculate entropy
            entropy = self._calculate_entropy(content)

            # Analyze entropy patterns
            analysis = self._analyze_entropy_pattern(content, entropy)

            # Determine threat score based on entropy
            score = self._calculate_threat_score(entropy, analysis)

            # Determine threat level
            if score >= 0.8:
                threat_level = ThreatLevel.HIGH
            elif score >= 0.6:
                threat_level = ThreatLevel.MEDIUM
            else:
                threat_level = ThreatLevel.LOW

            return SecurityResult(
                method=DetectionMethod.ENTROPY_ANALYSIS,
                score=score,
                threat_level=threat_level,
                confidence=min(1.0, len(content) / 1024),  # Higher confidence for larger files
                details={
                    "entropy": entropy,
                    "file_size": len(content),
                    "analysis": analysis,
                    "high_entropy_threshold": self.high_entropy_threshold,
                    "low_entropy_threshold": self.low_entropy_threshold
                }
            )

        except Exception as e:
            self.logger.error(f"Entropy analysis failed: {e}")
            return SecurityResult(
                method=DetectionMethod.ENTROPY_ANALYSIS,
                score=0.0,
                threat_level=ThreatLevel.LOW,
                confidence=0.0,
                details={"error": str(e)}
            )

    def _calculate_entropy(self, data: bytes) -> float:
        """
        Calculate Shannon entropy of byte data.

        Returns entropy value between 0 (no randomness) and 8 (maximum randomness).
        """
        if len(data) == 0:
            return 0.0

        # Count byte frequencies
        byte_counts = {}
        for byte in data:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1

        # Calculate entropy
        entropy = 0.0
        data_len = len(data)

        for count in byte_counts.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def _analyze_entropy_pattern(self, data: bytes, entropy: float) -> Dict[str, Any]:
        """Analyze entropy patterns for suspicious indicators."""
        analysis = {
            "entropy_level": "normal",
            "suspicious_patterns": [],
            "compression_indicators": [],
            "encryption_indicators": []
        }

        # Classify entropy level
        if entropy >= self.high_entropy_threshold:
            analysis["entropy_level"] = "high"
        elif entropy <= self.low_entropy_threshold:
            analysis["entropy_level"] = "low"

        # Check for encryption indicators
        if entropy >= self.high_entropy_threshold:
            analysis["encryption_indicators"].extend([
                "High entropy suggests encryption or compression",
                "Random byte distribution detected"
            ])

            # Check for specific encryption patterns
            if self._has_uniform_distribution(data):
                analysis["encryption_indicators"].append("Uniform byte distribution (strong encryption indicator)")

        # Check for compression indicators
        if entropy >= 6.0 and entropy < self.high_entropy_threshold:
            analysis["compression_indicators"].append("Moderate-high entropy suggests compression")

        # Check for suspicious patterns
        if self._has_suspicious_patterns(data):
            analysis["suspicious_patterns"].extend([
                "Contains suspicious byte patterns",
                "Potential packed executable indicators"
            ])

        # File type specific analysis
        file_extension = self._get_file_extension(data) if isinstance(data, str) else None
        if file_extension:
            expected_entropy = self._get_expected_entropy_for_type(file_extension)
            if expected_entropy and abs(entropy - expected_entropy) > 2.0:
                analysis["suspicious_patterns"].append(
                    f"Entropy deviation from expected for {file_extension} files"
                )

        return analysis

    def _has_uniform_distribution(self, data: bytes, threshold: float = 0.8) -> bool:
        """Check if byte distribution is suspiciously uniform."""
        if len(data) < 256:  # Need enough data for meaningful analysis
            return False

        byte_counts = {}
        for byte in data:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1

        expected_count = len(data) / 256
        uniform_bytes = 0

        for count in byte_counts.values():
            if abs(count - expected_count) / expected_count < 0.5:  # Within 50% of expected
                uniform_bytes += 1

        return (uniform_bytes / 256) > threshold

    def _has_suspicious_patterns(self, data: bytes) -> bool:
        """Check for suspicious byte patterns."""
        if len(data) < 4:
            return False

        # Check for common packer signatures or suspicious patterns
        suspicious_patterns = [
            b'\x60\xE8\x00\x00\x00\x00',  # UPX header
            b'MZ',  # PE file header
            b'\x7FELF',  # ELF header
            b'PK\x03\x04',  # ZIP header
            b'Rar!',  # RAR header
        ]

        for pattern in suspicious_patterns:
            if pattern in data[:min(1024, len(data))]:  # Check first 1KB
                return True

        return False

    def _get_file_extension(self, file_path: str) -> Optional[str]:
        """Extract file extension from path."""
        return Path(file_path).suffix.lower() if file_path else None

    def _get_expected_entropy_for_type(self, extension: str) -> Optional[float]:
        """Get expected entropy range for file type."""
        # Expected entropy values for common file types
        entropy_map = {
            '.txt': 4.0,    # Text files
            '.jpg': 7.5,    # Compressed images
            '.png': 7.8,    # Compressed images
            '.zip': 7.5,    # Compressed archives
            '.exe': 6.5,    # Executables
            '.dll': 6.0,    # Libraries
            '.pdf': 5.5,    # Documents
        }

        return entropy_map.get(extension)

    def _calculate_threat_score(self, entropy: float, analysis: Dict[str, Any]) -> float:
        """Calculate threat score based on entropy and analysis."""
        score = 0.0

        # Base score from entropy
        if entropy >= self.high_entropy_threshold:
            score += 0.7  # High entropy is suspicious
        elif entropy >= 6.0:
            score += 0.4  # Moderate-high entropy
        elif entropy <= self.low_entropy_threshold:
            score += 0.2  # Very low entropy might indicate obfuscation

        # Adjust based on analysis
        if analysis["entropy_level"] == "high":
            score += 0.2

        # Suspicious patterns increase score
        suspicious_count = len(analysis["suspicious_patterns"])
        score += min(0.3, suspicious_count * 0.1)

        # Encryption indicators
        encryption_count = len(analysis["encryption_indicators"])
        score += min(0.4, encryption_count * 0.15)

        return min(1.0, score)

    def is_available(self) -> bool:
        """Check if entropy analyzer is available."""
        return True  # Always available (uses only standard library)