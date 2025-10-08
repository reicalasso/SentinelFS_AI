"""
Content Inspection Engine

This module provides deep content analysis for files, including
pattern matching, metadata extraction, and behavioral indicators.
"""

import re
import os
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
import hashlib

from .engine import BaseDetector, DetectionMethod, SecurityResult, ThreatLevel


class ContentInspector(BaseDetector):
    """
    Advanced content inspection for file analysis.

    This detector examines file content for suspicious patterns,
    metadata anomalies, and behavioral indicators.
    """

    def __init__(self):
        super().__init__("Content Inspector", DetectionMethod.CONTENT_INSPECTION)
        self.suspicious_patterns = self._load_suspicious_patterns()
        self.safe_extensions = self._load_safe_extensions()

    def _load_suspicious_patterns(self) -> List[Dict[str, Any]]:
        """Load patterns that indicate suspicious content."""
        return [
            {
                "name": "powershell_obfuscation",
                "pattern": r"(?i)(IEX|Invoke-Expression)\s*\([^)]*\$\{[^}]*\}[^)]*\)",
                "score": 0.8,
                "description": "PowerShell code obfuscation"
            },
            {
                "name": "base64_encoded",
                "pattern": r"[A-Za-z0-9+/]{50,}=*",
                "score": 0.6,
                "description": "Long base64 encoded content"
            },
            {
                "name": "suspicious_urls",
                "pattern": r"(?i)(http|https|ftp)://[^\s]*\.(onion|tor|dark)",
                "score": 0.9,
                "description": "Dark web URLs"
            },
            {
                "name": "command_injection",
                "pattern": r"(?i)(cmd\.exe|powershell\.exe|/bin/bash|/bin/sh)\s+[^;\n]*[;&|]",
                "score": 0.7,
                "description": "Command injection patterns"
            },
            {
                "name": "ransomware_notes",
                "pattern": r"(?i)(your files are encrypted|pay bitcoin|decryptor)",
                "score": 0.9,
                "description": "Ransomware ransom notes"
            },
            {
                "name": "malicious_macros",
                "pattern": r"(?i)(AutoOpen|AutoClose|Document_Open)\s*\(",
                "score": 0.6,
                "description": "Malicious Office macros"
            },
            {
                "name": "suspicious_imports",
                "pattern": r"(?i)(import\s+(os|subprocess|socket|urllib|requests)\s*;)",
                "score": 0.4,
                "description": "Suspicious Python imports"
            }
        ]

    def _load_safe_extensions(self) -> Set[str]:
        """Load extensions considered generally safe."""
        return {
            '.txt', '.md', '.rst', '.pdf', '.doc', '.docx', '.xls', '.xlsx',
            '.ppt', '.pptx', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
            '.mp3', '.mp4', '.avi', '.wav', '.zip', '.rar', '.7z', '.tar.gz'
        }

    def analyze(self, data: Any, context: Optional[Dict[str, Any]] = None) -> SecurityResult:
        """
        Analyze file content for suspicious patterns.

        Args:
            data: File path (str) or file content (bytes/str)
            context: Additional context information

        Returns:
            SecurityResult with content analysis
        """
        try:
            # Get file content and metadata
            content, metadata = self._extract_content_and_metadata(data)

            if not content:
                return SecurityResult(
                    method=DetectionMethod.CONTENT_INSPECTION,
                    score=0.0,
                    threat_level=ThreatLevel.LOW,
                    confidence=0.0,
                    details={"error": "No content to analyze"}
                )

            # Perform content analysis
            analysis_results = self._analyze_content(content, metadata)

            # Calculate overall threat score
            score = self._calculate_content_score(analysis_results)

            # Determine threat level
            if score >= 0.8:
                threat_level = ThreatLevel.CRITICAL
            elif score >= 0.6:
                threat_level = ThreatLevel.HIGH
            elif score >= 0.4:
                threat_level = ThreatLevel.MEDIUM
            else:
                threat_level = ThreatLevel.LOW

            return SecurityResult(
                method=DetectionMethod.CONTENT_INSPECTION,
                score=score,
                threat_level=threat_level,
                confidence=min(1.0, len(content) / 1000),  # Higher confidence for larger content
                details={
                    "content_length": len(content),
                    "patterns_found": analysis_results["patterns_found"],
                    "metadata": metadata,
                    "analysis": analysis_results
                }
            )

        except Exception as e:
            self.logger.error(f"Content inspection failed: {e}")
            return SecurityResult(
                method=DetectionMethod.CONTENT_INSPECTION,
                score=0.0,
                threat_level=ThreatLevel.LOW,
                confidence=0.0,
                details={"error": str(e)}
            )

    def _extract_content_and_metadata(self, data: Any) -> Tuple[str, Dict[str, Any]]:
        """Extract content and metadata from input data."""
        metadata = {}

        if isinstance(data, str) and os.path.isfile(data):
            # File path provided
            file_path = Path(data)
            metadata["file_path"] = str(file_path)
            metadata["file_name"] = file_path.name
            metadata["file_extension"] = file_path.suffix.lower()
            metadata["file_size"] = file_path.stat().st_size

            # Read content
            try:
                with open(data, 'rb') as f:
                    raw_content = f.read()

                # Try to decode as text
                try:
                    content = raw_content.decode('utf-8', errors='ignore')
                except UnicodeDecodeError:
                    content = str(raw_content[:1024])  # First 1KB as string

                metadata["is_text"] = self._is_text_file(raw_content)

            except Exception as e:
                content = ""
                metadata["read_error"] = str(e)

        elif isinstance(data, bytes):
            # Raw bytes provided
            content = data.decode('utf-8', errors='ignore')
            metadata["file_size"] = len(data)
            metadata["is_text"] = self._is_text_file(data)

        elif isinstance(data, str):
            # String content provided
            content = data
            metadata["file_size"] = len(data)
            metadata["is_text"] = True

        else:
            content = ""
            metadata["error"] = "Unsupported data type"

        return content, metadata

    def _is_text_file(self, data: bytes, threshold: float = 0.7) -> bool:
        """Determine if file content is primarily text."""
        if len(data) == 0:
            return True

        # Count printable characters
        printable = sum(1 for byte in data if 32 <= byte <= 126 or byte in (9, 10, 13))
        ratio = printable / len(data)

        return ratio >= threshold

    def _analyze_content(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content for suspicious patterns."""
        results = {
            "patterns_found": [],
            "pattern_matches": [],
            "content_analysis": {},
            "metadata_analysis": {}
        }

        # Check for suspicious patterns
        for pattern_info in self.suspicious_patterns:
            matches = re.findall(pattern_info["pattern"], content, re.IGNORECASE)
            if matches:
                results["patterns_found"].append(pattern_info["name"])
                results["pattern_matches"].append({
                    "pattern": pattern_info["name"],
                    "description": pattern_info["description"],
                    "score": pattern_info["score"],
                    "matches": len(matches),
                    "examples": matches[:3]  # First 3 matches
                })

        # Analyze content characteristics
        results["content_analysis"] = self._analyze_content_characteristics(content)

        # Analyze metadata
        results["metadata_analysis"] = self._analyze_metadata(metadata)

        return results

    def _analyze_content_characteristics(self, content: str) -> Dict[str, Any]:
        """Analyze general content characteristics."""
        analysis = {
            "has_executable_content": False,
            "has_suspicious_keywords": False,
            "compression_indicators": False,
            "obfuscation_indicators": False
        }

        # Check for executable content indicators
        exec_patterns = [
            r"(?i)(eval|exec|system|shell_exec|popen)",
            r"(?i)(subprocess|os\.system|os\.popen)",
            r"(?i)(import\s+(os|sys|subprocess))"
        ]

        for pattern in exec_patterns:
            if re.search(pattern, content):
                analysis["has_executable_content"] = True
                break

        # Check for suspicious keywords
        suspicious_keywords = [
            "password", "credential", "hack", "exploit", "malware",
            "virus", "trojan", "backdoor", "rootkit", "keylogger"
        ]

        for keyword in suspicious_keywords:
            if keyword.lower() in content.lower():
                analysis["has_suspicious_keywords"] = True
                break

        # Check for compression indicators
        if re.search(r"(?i)(gzip|zip|rar|7z|bzip2|xz)", content):
            analysis["compression_indicators"] = True

        # Check for obfuscation indicators
        obfuscation_patterns = [
            r"[A-Za-z0-9+/]{100,}=*",  # Very long base64
            r"(?i)(from\s+base64\s+import|base64\.b64decode)",
            r"\$\{[^\}]+\}",  # Variable substitution
        ]

        for pattern in obfuscation_patterns:
            if re.search(pattern, content):
                analysis["obfuscation_indicators"] = True
                break

        return analysis

    def _analyze_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze file metadata for suspicious indicators."""
        analysis = {
            "extension_risk": "low",
            "size_anomaly": False,
            "name_suspicious": False
        }

        # Check file extension
        extension = metadata.get("file_extension", "")
        if extension:
            if extension not in self.safe_extensions:
                analysis["extension_risk"] = "medium"
                # Check for double extensions
                if extension.count('.') > 0:
                    analysis["extension_risk"] = "high"

        # Check file size anomalies
        size = metadata.get("file_size", 0)
        if size > 100 * 1024 * 1024:  # > 100MB
            analysis["size_anomaly"] = True

        # Check filename for suspicious patterns
        filename = metadata.get("file_name", "")
        if filename:
            suspicious_name_patterns = [
                r"(?i)(password|secret|key|credential)",
                r"(?i)(hack|exploit|malware|virus)",
                r"[0-9]{8,}",  # Long numbers
                r"(?i)(temp|tmp)[0-9]+",
            ]

            for pattern in suspicious_name_patterns:
                if re.search(pattern, filename):
                    analysis["name_suspicious"] = True
                    break

        return analysis

    def _calculate_content_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall threat score from content analysis."""
        score = 0.0

        # Pattern-based scoring
        for match in analysis_results["pattern_matches"]:
            score += match["score"] * min(1.0, match["matches"] / 5)  # Cap per pattern

        # Content analysis scoring
        content_analysis = analysis_results["content_analysis"]
        if content_analysis["has_executable_content"]:
            score += 0.4
        if content_analysis["has_suspicious_keywords"]:
            score += 0.2
        if content_analysis["compression_indicators"]:
            score += 0.1
        if content_analysis["obfuscation_indicators"]:
            score += 0.3

        # Metadata analysis scoring
        metadata_analysis = analysis_results["metadata_analysis"]
        if metadata_analysis["extension_risk"] == "high":
            score += 0.3
        elif metadata_analysis["extension_risk"] == "medium":
            score += 0.1

        if metadata_analysis["size_anomaly"]:
            score += 0.2
        if metadata_analysis["name_suspicious"]:
            score += 0.2

        return min(1.0, score)

    def is_available(self) -> bool:
        """Check if content inspector is available."""
        return True  # Always available (uses standard library)