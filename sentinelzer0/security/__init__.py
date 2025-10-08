"""
Security Engine Module for SentinelFS AI

This module provides advanced security capabilities including:
- YARA rule engine integration for signature-based detection
- Entropy analysis for encryption detection
- Multi-layered threat scoring and correlation
- Content inspection hooks for file analysis

The security engine works alongside the AI model to provide
comprehensive threat detection with multiple detection methods.
"""

from .engine import SecurityEngine
from .yara_detector import YaraDetector
from .entropy_analyzer import EntropyAnalyzer
from .threat_correlator import ThreatCorrelator
from .content_inspector import ContentInspector

__all__ = [
    'SecurityEngine',
    'YaraDetector',
    'EntropyAnalyzer',
    'ThreatCorrelator',
    'ContentInspector'
]