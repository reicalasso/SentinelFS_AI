"""
YARA Rule Engine Integration

This module provides integration with YARA (Yet Another Recursive Acronym)
for signature-based malware detection.
"""

import os
import tempfile
from typing import Dict, List, Optional, Any
from pathlib import Path

from .engine import BaseDetector, DetectionMethod, SecurityResult, ThreatLevel

try:
    import yara
    YARA_AVAILABLE = True
except ImportError:
    YARA_AVAILABLE = False
    yara = None


class YaraDetector(BaseDetector):
    """
    YARA-based signature detection for known malware patterns.

    This detector uses YARA rules to identify known malicious patterns
    in file content and metadata.
    """

    def __init__(self, rules_path: Optional[str] = None):
        super().__init__("YARA Detector", DetectionMethod.YARA_SIGNATURES)
        self.rules_path = rules_path or self._get_default_rules_path()
        self.compiled_rules = None
        self._load_rules()

    def _get_default_rules_path(self) -> str:
        """Get default path for YARA rules."""
        module_dir = Path(__file__).parent
        rules_dir = module_dir / "rules"
        rules_dir.mkdir(exist_ok=True)
        return str(rules_dir)

    def _load_rules(self):
        """Load and compile YARA rules."""
        if not YARA_AVAILABLE:
            self.logger.warning("YARA library not available")
            return

        rules_dir = Path(self.rules_path)
        if not rules_dir.exists():
            self.logger.info(f"Creating YARA rules directory: {rules_dir}")
            rules_dir.mkdir(parents=True, exist_ok=True)
            self._create_default_rules()
            return

        rule_files = list(rules_dir.glob("*.yar"))
        if not rule_files:
            self.logger.info("No YARA rule files found, creating defaults")
            self._create_default_rules()
            rule_files = list(rules_dir.glob("*.yar"))

        if rule_files:
            try:
                # Compile all rules
                rule_dict = {}
                for rule_file in rule_files:
                    namespace = rule_file.stem
                    with open(rule_file, 'r') as f:
                        rule_content = f.read()
                    rule_dict[namespace] = rule_content

                self.compiled_rules = yara.compile(sources=rule_dict)
                self.logger.info(f"Loaded {len(rule_files)} YARA rule files")
            except Exception as e:
                self.logger.error(f"Failed to compile YARA rules: {e}")
        else:
            self.logger.warning("No YARA rules available")

    def _create_default_rules(self):
        """Create default YARA rules for common threats."""
        rules_dir = Path(self.rules_path)

        # Ransomware patterns
        ransomware_rules = '''
rule Ransomware_Extensions {
    meta:
        description = "Common ransomware file extensions"
        threat_level = "high"
    strings:
        $ext1 = ".encrypted" nocase
        $ext2 = ".locked" nocase
        $ext3 = ".crypto" nocase
        $ext4 = ".crypt" nocase
        $ext5 = ".readme" nocase
    condition:
        any of them
}

rule Suspicious_Processes {
    meta:
        description = "Suspicious process names"
        threat_level = "medium"
    strings:
        $proc1 = "vssadmin.exe" nocase
        $proc2 = "bcdedit.exe" nocase
        $proc3 = "net.exe" nocase
        $proc4 = "sc.exe" nocase
    condition:
        any of them
}
'''

        # Malware patterns
        malware_rules = '''
rule Malware_Patterns {
    meta:
        description = "Common malware byte patterns"
        threat_level = "high"
    strings:
        $mz = { 4D 5A }  // MZ header
        $suspicious1 = { 90 90 90 90 }  // NOP sled
        $suspicious2 = { CC CC CC CC }  // INT3 breakpoints
    condition:
        $mz at 0 and any of ($suspicious*)
}
'''

        # Write rules to files
        with open(rules_dir / "ransomware.yar", 'w') as f:
            f.write(ransomware_rules)

        with open(rules_dir / "malware.yar", 'w') as f:
            f.write(malware_rules)

        self.logger.info("Created default YARA rules")

    def analyze(self, data: Any, context: Optional[Dict[str, Any]] = None) -> SecurityResult:
        """
        Analyze data using YARA rules.

        Args:
            data: File path (str) or file content (bytes)
            context: Additional context information

        Returns:
            SecurityResult with YARA analysis
        """
        if not self.is_available():
            return SecurityResult(
                method=DetectionMethod.YARA_SIGNATURES,
                score=0.0,
                threat_level=ThreatLevel.LOW,
                confidence=0.0,
                details={"error": "YARA not available"}
            )

        matched_rules = []
        max_score = 0.0

        try:
            # Prepare data for scanning
            if isinstance(data, str) and os.path.isfile(data):
                # Scan file directly
                matches = self.compiled_rules.match(data)
            elif isinstance(data, bytes):
                # Scan bytes in memory
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(data)
                    tmp.flush()
                    matches = self.compiled_rules.match(tmp.name)
                    os.unlink(tmp.name)
            else:
                # Invalid data type
                return SecurityResult(
                    method=DetectionMethod.YARA_SIGNATURES,
                    score=0.0,
                    threat_level=ThreatLevel.LOW,
                    confidence=0.0,
                    details={"error": "Invalid data type for YARA analysis"}
                )

            # Process matches
            for match in matches:
                matched_rules.append(match.rule)
                # Extract threat level from rule metadata
                threat_score = self._get_rule_threat_score(match)
                max_score = max(max_score, threat_score)

            # Determine overall threat level
            if max_score >= 0.8:
                threat_level = ThreatLevel.CRITICAL
            elif max_score >= 0.6:
                threat_level = ThreatLevel.HIGH
            elif max_score >= 0.4:
                threat_level = ThreatLevel.MEDIUM
            else:
                threat_level = ThreatLevel.LOW

            return SecurityResult(
                method=DetectionMethod.YARA_SIGNATURES,
                score=max_score,
                threat_level=threat_level,
                confidence=min(1.0, len(matched_rules) * 0.3),  # Confidence based on number of matches
                details={
                    "matched_rules": matched_rules,
                    "total_matches": len(matched_rules),
                    "rules_compiled": self.compiled_rules is not None
                },
                matched_rules=matched_rules
            )

        except Exception as e:
            self.logger.error(f"YARA analysis failed: {e}")
            return SecurityResult(
                method=DetectionMethod.YARA_SIGNATURES,
                score=0.0,
                threat_level=ThreatLevel.LOW,
                confidence=0.0,
                details={"error": str(e)}
            )

    def _get_rule_threat_score(self, match) -> float:
        """Extract threat score from YARA rule match."""
        # Check rule metadata for threat level
        if hasattr(match, 'meta') and 'threat_level' in match.meta:
            threat_level = match.meta['threat_level'].lower()
            if threat_level == 'critical':
                return 1.0
            elif threat_level == 'high':
                return 0.8
            elif threat_level == 'medium':
                return 0.6
            elif threat_level == 'low':
                return 0.3

        # Default score based on rule matching
        return 0.5

    def is_available(self) -> bool:
        """Check if YARA is available and rules are loaded."""
        return YARA_AVAILABLE and self.compiled_rules is not None

    def add_rule(self, rule_name: str, rule_content: str):
        """Add a new YARA rule dynamically."""
        if not YARA_AVAILABLE:
            raise RuntimeError("YARA library not available")

        try:
            # Compile new rule
            new_rules = yara.compile(source=rule_content)

            # Merge with existing rules if any
            if self.compiled_rules:
                # Note: YARA doesn't support dynamic rule addition easily
                # This is a simplified implementation
                self.logger.warning("Dynamic rule addition not fully supported")
            else:
                self.compiled_rules = new_rules

            self.logger.info(f"Added YARA rule: {rule_name}")

        except Exception as e:
            self.logger.error(f"Failed to add YARA rule {rule_name}: {e}")
            raise