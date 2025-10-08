#!/usr/bin/env python3
"""
Test Phase 1.3: Production Monitoring

This script tests all monitoring components to ensure they work correctly.
"""

import sys
import os
import time
import requests
import json
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_monitoring_components():
    """Test all monitoring components."""
    print("🧪 Testing Phase 1.3: Production Monitoring")
    print("=" * 50)

    # Test 1: Import monitoring components
    print("\n1. Testing imports...")
    try:
        from sentinelzer0.monitoring.metrics import init_metrics, REQUEST_COUNT
        from sentinelzer0.monitoring.drift_detector import ModelDriftDetector
        from sentinelzer0.monitoring.alerts import AlertManager
        from sentinelzer0.monitoring.logging_config import setup_logging
        print("✅ All monitoring components imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        raise AssertionError("Test failed")

    # Test 2: Initialize metrics
    print("\n2. Testing metrics initialization...")
    try:
        init_metrics(
            model_name="test_model",
            model_version="1.0.0",
            model_type="test"
        )
        print("✅ Metrics initialized successfully")
    except Exception as e:
        print(f"❌ Metrics initialization failed: {e}")
        raise AssertionError("Test failed")

    # Test 3: Drift detector
    print("\n3. Testing drift detection...")
    try:
        detector = ModelDriftDetector(window_size=100)
        baseline_scores = [0.1, 0.2, 0.15, 0.25, 0.18] * 20  # 100 scores
        detector.set_baseline(baseline_scores)

        # Add some predictions
        for score in [0.12, 0.18, 0.22, 0.15, 0.19]:
            detector.add_prediction(score)

        status = detector.get_drift_status()
        print(f"✅ Drift detector working - Baseline set: {status['baseline_set']}")
    except Exception as e:
        print(f"❌ Drift detector failed: {e}")
        raise AssertionError("Test failed")

    # Test 4: Alert manager
    print("\n4. Testing alert system...")
    try:
        alert_manager = AlertManager()

        # Check for alerts with normal metrics
        metrics = {'avg_latency': 0.5, 'error_rate': 0.01, 'drift_score': 0.02}
        alert_manager.check_alerts(metrics)

        alerts = alert_manager.get_active_alerts()
        print(f"✅ Alert manager working - Active alerts: {len(alerts)}")
    except Exception as e:
        print(f"❌ Alert manager failed: {e}")
        raise AssertionError("Test failed")

    # Test 5: Structured logging
    print("\n5. Testing structured logging...")
    try:
        logger = setup_logging(level="INFO")
        logger.log_performance("test_operation", 1.5, {"test": "data"})
        logger.log_request("GET", "/test", 200, 45.2)
        print("✅ Structured logging working")
    except Exception as e:
        print(f"❌ Structured logging failed: {e}")
        raise AssertionError("Test failed")

    # Test 6: Check if Grafana dashboard exists
    print("\n6. Testing Grafana dashboard...")
    dashboard_path = Path("sentinelzer0/monitoring/grafana/dashboard.json")
    if dashboard_path.exists():
        with open(dashboard_path) as f:
            dashboard = json.load(f)
        if dashboard.get("dashboard", {}).get("title") == "SentinelFS AI - Production Monitoring":
            print("✅ Grafana dashboard configuration found")
        else:
            print("❌ Grafana dashboard configuration invalid")
            raise AssertionError("Test failed")
    else:
        print("❌ Grafana dashboard file not found")
        raise AssertionError("Test failed")

    # Test 7: Check setup script
    print("\n7. Testing setup script...")
    setup_script = Path("setup_monitoring.sh")
    if setup_script.exists() and setup_script.stat().st_mode & 0o111:
        print("✅ Monitoring setup script found and executable")
    else:
        print("❌ Setup script not found or not executable")
        raise AssertionError("Test failed")

    print("\n" + "=" * 50)
    print("🎉 All Phase 1.3 monitoring components tested successfully!")
    print("\nNext steps:")
    print("1. Start the API: python sentinelzer0/api/server.py")
    print("2. Setup monitoring: ./setup_monitoring.sh")
    print("3. Access dashboards:")
    print("   - API: http://localhost:8000/docs")
    print("   - Prometheus: http://localhost:9090")
    print("   - Grafana: http://localhost:3000")

    assert True

if __name__ == "__main__":
    success = test_monitoring_components()
    sys.exit(0 if success else 1)