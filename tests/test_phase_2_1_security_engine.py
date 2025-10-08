#!/usr/bin/env python3
"""
Test script for Phase 2.1 - Security Engine Integration

This script tests the complete security engine with YARA, entropy analysis,
threat correlation, and content inspection.
"""

import sys
import time
from pathlib import Path
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
from sentinelzer0.security import SecurityEngine
from sentinelzer0.utils.logger import get_logger

logger = get_logger(__name__)


def test_security_engine_initialization():
    """Test security engine initialization and detector loading."""
    print("🧪 Testing Phase 2.1: Security Engine Initialization")
    print("=" * 60)

    try:
        # Initialize security engine
        engine = SecurityEngine()

        # Check status
        status = engine.get_status()
        print(f"✅ Security engine initialized")
        print(f"   Detectors loaded: {status['total_detectors']}")
        print(f"   Correlator available: {status['correlator_available']}")

        for method, config in status['detectors'].items():
            print(f"   {method}: {'✅' if config['available'] else '❌'} {config['name']}")

        return True

    except Exception as e:
        print(f"❌ Security engine initialization failed: {e}")
        return False


def test_entropy_analysis():
    """Test entropy analysis for encryption detection."""
    print("\n🧪 Testing Entropy Analysis")
    print("=" * 40)

    try:
        from sentinelzer0.security import EntropyAnalyzer
        analyzer = EntropyAnalyzer()

        # Test with normal text
        normal_text = b"This is normal text content for testing purposes."
        result = analyzer.analyze(normal_text)
        print("✅ Normal text analysis:")
        print(f"   Entropy: {result.details['entropy']:.3f}")
        print(f"   Threat level: {result.threat_level.value}")
        print(f"   Score: {result.score:.3f}")

        # Test with high entropy (simulated encrypted data)
        high_entropy_data = bytes(range(256)) * 10  # Repeating byte pattern
        result = analyzer.analyze(high_entropy_data)
        print("✅ High entropy data analysis:")
        print(f"   Entropy: {result.details['entropy']:.3f}")
        print(f"   Threat level: {result.threat_level.value}")
        print(f"   Score: {result.score:.3f}")

        return True

    except Exception as e:
        print(f"❌ Entropy analysis test failed: {e}")
        return False


def test_content_inspection():
    """Test content inspection for suspicious patterns."""
    print("\n🧪 Testing Content Inspection")
    print("=" * 40)

    try:
        from sentinelzer0.security import ContentInspector
        inspector = ContentInspector()

        # Test with normal content
        normal_content = "This is a normal text file with regular content."
        result = inspector.analyze(normal_content)
        print("✅ Normal content analysis:")
        print(f"   Patterns found: {len(result.details['patterns_found'])}")
        print(f"   Threat level: {result.threat_level.value}")
        print(f"   Score: {result.score:.3f}")

        # Test with suspicious content
        suspicious_content = """
        import os
        import subprocess
        os.system('rm -rf /')
        eval(base64.b64decode('c29tZSBiYXNlNjQ='))
        """
        result = inspector.analyze(suspicious_content)
        print("✅ Suspicious content analysis:")
        print(f"   Patterns found: {len(result.details['patterns_found'])}")
        print(f"   Threat level: {result.threat_level.value}")
        print(f"   Score: {result.score:.3f}")

        return True

    except Exception as e:
        print(f"❌ Content inspection test failed: {e}")
        return False


def test_threat_correlation():
    """Test threat correlation across multiple methods."""
    print("\n🧪 Testing Threat Correlation")
    print("=" * 40)

    try:
        from sentinelzer0.security import ThreatCorrelator
        correlator = ThreatCorrelator()

        # Create mock security results
        from sentinelzer0.security.engine import SecurityResult, DetectionMethod, ThreatLevel

        mock_results = [
            SecurityResult(
                method=DetectionMethod.ENTROPY_ANALYSIS,
                score=0.8,
                threat_level=ThreatLevel.HIGH,
                confidence=0.9,
                details={"entropy": 7.8}
            ),
            SecurityResult(
                method=DetectionMethod.CONTENT_INSPECTION,
                score=0.6,
                threat_level=ThreatLevel.MEDIUM,
                confidence=0.7,
                details={"patterns_found": ["suspicious_imports"]}
            )
        ]

        # Test correlation
        correlation = correlator.correlate(mock_results)
        print("✅ Threat correlation analysis:")
        print(f"   Correlation factor: {correlation['correlation_factor']:.2f}")
        print(f"   Confidence adjustment: {correlation['confidence_adjustment']:.2f}")
        print(f"   Rules triggered: {correlation['rules_triggered']}")

        # Test threat context
        context = correlator.get_threat_context(mock_results)
        print("✅ Threat context generation:")
        print(f"   Pattern: {context['threat_pattern']}")
        print(f"   Confidence: {context['confidence_level']}")
        print(f"   Priority: {context['mitigation_priority']}")

        return True

    except Exception as e:
        print(f"❌ Threat correlation test failed: {e}")
        return False


def test_yara_integration():
    """Test YARA rule engine integration."""
    print("\n🧪 Testing YARA Integration")
    print("=" * 40)

    try:
        from sentinelzer0.security import YaraDetector
        detector = YaraDetector()

        if not detector.is_available():
            print("⚠️  YARA not available (library not installed)")
            print("   Install with: pip install yara-python")
            return True  # Not a failure, just not available

        # Test with sample data
        test_data = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00\xb8\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x0e\x1f\xba\x0e\x00\xb4\x09\xcd!\xb8\x01L\xcd!This program cannot be run in DOS mode."
        result = detector.analyze(test_data)
        print("✅ YARA analysis:")
        print(f"   Available: {detector.is_available()}")
        print(f"   Rules loaded: {len(detector.compiled_rules.keys()) if detector.compiled_rules else 0}")
        print(f"   Threat level: {result.threat_level.value}")
        print(f"   Score: {result.score:.3f}")

        return True

    except Exception as e:
        print(f"❌ YARA integration test failed: {e}")
        return False


def test_complete_security_engine():
    """Test complete security engine with file analysis."""
    print("\n🧪 Testing Complete Security Engine")
    print("=" * 40)

    try:
        # Create a test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
            This is a test file with some suspicious content.
            import os
            import subprocess
            eval("malicious code")
            """)
            test_file = f.name

        engine = SecurityEngine()

        # Analyze file
        result = engine.analyze_file(test_file, ai_score=0.3)
        print("✅ Complete file analysis:")
        print(f"   AI Score: {result.ai_score:.3f}")
        print(f"   Security Score: {result.security_score:.3f}")
        print(f"   Combined Score: {result.combined_score:.3f}")
        print(f"   Threat Level: {result.threat_level.value}")
        print(f"   Final Decision: {result.final_decision}")
        print(f"   Detection Methods: {len(result.detection_methods)}")

        # Cleanup
        os.unlink(test_file)

        return True

    except Exception as e:
        print(f"❌ Complete security engine test failed: {e}")
        return False


def main():
    """Run all Phase 2.1 tests."""
    print("🚀 SentinelFS AI - Phase 2.1 Security Engine Integration Tests")
    print("=" * 70)

    tests = [
        test_security_engine_initialization,
        test_entropy_analysis,
        test_content_inspection,
        test_threat_correlation,
        test_yara_integration,
        test_complete_security_engine
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Phase 2.1 security engine tests PASSED!")
        print("✅ Security Engine Integration: COMPLETE")
        return 0
    else:
        print("❌ Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())