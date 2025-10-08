#!/usr/bin/env python3
"""
Test script for Phase 1.2 - REST API Framework

This script tests the FastAPI server with comprehensive endpoint validation.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import requests
from sentinelzer0.utils.logger import get_logger

logger = get_logger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "sentinelfs-dev-key-2025"
HEADERS = {"X-API-Key": API_KEY}


def test_root_endpoint():
    """Test root endpoint (no auth required)."""
    logger.info("=" * 80)
    logger.info("TEST 1: Root Endpoint")
    logger.info("=" * 80)
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["status"] == "operational"
        
        logger.info(f"✓ API Name: {data['name']}")
        logger.info(f"✓ Version: {data['version']}")
        logger.info(f"✓ Status: {data['status']}")
        logger.info("✅ Root endpoint test passed!\n")
        assert True
        
    except Exception as e:
        logger.error(f"❌ Root endpoint test failed: {e}")
        raise AssertionError("Test failed")


def test_health_endpoint():
    """Test health check endpoint."""
    logger.info("=" * 80)
    logger.info("TEST 2: Health Check Endpoint")
    logger.info("=" * 80)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "gpu_available" in data
        
        logger.info(f"✓ Status: {data['status']}")
        logger.info(f"✓ Model Loaded: {data['model_loaded']}")
        logger.info(f"✓ GPU Available: {data['gpu_available']}")
        logger.info(f"✓ Uptime: {data['uptime_seconds']:.2f}s")
        logger.info("✅ Health check test passed!\n")
        assert True
        
    except Exception as e:
        logger.error(f"❌ Health check test failed: {e}")
        raise AssertionError("Test failed")


def test_authentication():
    """Test API authentication."""
    logger.info("=" * 80)
    logger.info("TEST 3: Authentication")
    logger.info("=" * 80)
    
    try:
        # Test without API key
        response = requests.get(f"{API_BASE_URL}/metrics")
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        logger.info("✓ Request without API key rejected (401)")
        
        # Test with invalid API key
        bad_headers = {"X-API-Key": "invalid-key"}
        response = requests.get(f"{API_BASE_URL}/metrics", headers=bad_headers)
        assert response.status_code == 403, f"Expected 403, got {response.status_code}"
        logger.info("✓ Request with invalid API key rejected (403)")
        
        # Test with valid API key
        response = requests.get(f"{API_BASE_URL}/metrics", headers=HEADERS)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        logger.info("✓ Request with valid API key accepted (200)")
        
        logger.info("✅ Authentication test passed!\n")
        assert True
        
    except Exception as e:
        logger.error(f"❌ Authentication test failed: {e}")
        raise AssertionError("Test failed")


def test_metrics_endpoint():
    """Test metrics endpoint."""
    logger.info("=" * 80)
    logger.info("TEST 4: Metrics Endpoint")
    logger.info("=" * 80)
    
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", headers=HEADERS)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        required_fields = [
            'total_predictions', 'threats_detected', 'threat_rate',
            'avg_latency_ms', 'p50_latency_ms', 'p95_latency_ms',
            'p99_latency_ms', 'max_latency_ms', 'buffer_size'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        logger.info(f"✓ Total Predictions: {data['total_predictions']}")
        logger.info(f"✓ Threats Detected: {data['threats_detected']}")
        logger.info(f"✓ Threat Rate: {data['threat_rate']:.2f}%")
        logger.info(f"✓ Average Latency: {data['avg_latency_ms']:.2f}ms")
        logger.info(f"✓ Buffer Size: {data['buffer_size']}")
        logger.info("✅ Metrics endpoint test passed!\n")
        assert True
        
    except Exception as e:
        logger.error(f"❌ Metrics endpoint test failed: {e}")
        raise AssertionError("Test failed")


def test_model_info_endpoint():
    """Test model info endpoint."""
    logger.info("=" * 80)
    logger.info("TEST 5: Model Info Endpoint")
    logger.info("=" * 80)
    
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", headers=HEADERS)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "model_name" in data
        assert "version" in data
        assert "architecture" in data
        
        logger.info(f"✓ Model Name: {data['model_name']}")
        logger.info(f"✓ Version: {data['version']}")
        logger.info(f"✓ Input Features: {data['input_features']}")
        logger.info(f"✓ Sequence Length: {data['sequence_length']}")
        logger.info(f"✓ Threshold: {data['threshold']}")
        logger.info(f"✓ Architecture: {data['architecture']['type']}")
        logger.info("✅ Model info endpoint test passed!\n")
        assert True
        
    except Exception as e:
        logger.error(f"❌ Model info endpoint test failed: {e}")
        raise AssertionError("Test failed")


def test_predict_endpoint_single():
    """Test prediction endpoint with single event."""
    logger.info("=" * 80)
    logger.info("TEST 6: Prediction Endpoint (Single Event)")
    logger.info("=" * 80)
    
    try:
        # Create test event
        payload = {
            "events": [
                {
                    "event_type": "CREATE",
                    "path": "/tmp/test_file.txt",
                    "timestamp": time.time(),
                    "size": 1024,
                    "is_directory": False,
                    "user": "testuser",
                    "process": "python3",
                    "extension": ".txt"
                }
            ],
            "return_components": True
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            headers=HEADERS
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "predictions" in data
        assert "total_events" in data
        assert data["total_events"] == 1
        
        prediction = data["predictions"][0] if data["predictions"] else None
        if prediction:
            logger.info(f"✓ Event ID: {prediction['event_id']}")
            logger.info(f"✓ Threat Score: {prediction['threat_score']:.4f}")
            logger.info(f"✓ Is Threat: {prediction['is_threat']}")
            logger.info(f"✓ Confidence: {prediction['confidence']:.4f}")
            logger.info(f"✓ Latency: {prediction['latency_ms']:.2f}ms")
            if prediction.get('components'):
                logger.info(f"✓ Components: {prediction['components']}")
        
        logger.info(f"✓ Total Latency: {data['total_latency_ms']:.2f}ms")
        logger.info("✅ Single event prediction test passed!\n")
        assert True
        
    except Exception as e:
        logger.error(f"❌ Single event prediction test failed: {e}")
        raise AssertionError("Test failed")


def test_predict_endpoint_batch():
    """Test prediction endpoint with batch of events."""
    logger.info("=" * 80)
    logger.info("TEST 7: Prediction Endpoint (Batch)")
    logger.info("=" * 80)
    
    try:
        # Create batch of test events
        events = []
        for i in range(100):
            events.append({
                "event_type": "MODIFY" if i % 3 == 0 else "CREATE",
                "path": f"/tmp/test_file_{i}.txt",
                "timestamp": time.time(),
                "size": 1024 * (i + 1),
                "is_directory": False,
                "user": "testuser",
                "process": "python3",
                "extension": ".txt"
            })
        
        payload = {
            "events": events,
            "return_components": False
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            headers=HEADERS
        )
        end_time = time.time()
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data["total_events"] == 100
        
        request_time = (end_time - start_time) * 1000
        throughput = data["total_events"] / (request_time / 1000)
        
        logger.info(f"✓ Total Events: {data['total_events']}")
        logger.info(f"✓ Predictions: {len(data['predictions'])}")
        logger.info(f"✓ Threats Detected: {data['threats_detected']}")
        logger.info(f"✓ Total Latency: {data['total_latency_ms']:.2f}ms")
        logger.info(f"✓ Request Time: {request_time:.2f}ms")
        logger.info(f"✓ Throughput: {throughput:.2f} events/sec")
        logger.info("✅ Batch prediction test passed!\n")
        assert True
        
    except Exception as e:
        logger.error(f"❌ Batch prediction test failed: {e}")
        raise AssertionError("Test failed")


def test_config_endpoint():
    """Test configuration endpoint."""
    logger.info("=" * 80)
    logger.info("TEST 8: Configuration Endpoint")
    logger.info("=" * 80)
    
    try:
        # Update configuration
        payload = {
            "sequence_length": 64,
            "threshold": 0.6,
            "min_confidence": 0.7
        }
        
        response = requests.post(
            f"{API_BASE_URL}/config/stream",
            json=payload,
            headers=HEADERS
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data["success"] == True
        assert "config" in data
        
        logger.info(f"✓ Success: {data['success']}")
        logger.info(f"✓ Message: {data['message']}")
        logger.info(f"✓ New Threshold: {data['config']['threshold']}")
        logger.info(f"✓ New Min Confidence: {data['config']['min_confidence']}")
        logger.info("✅ Configuration endpoint test passed!\n")
        assert True
        
    except Exception as e:
        logger.error(f"❌ Configuration endpoint test failed: {e}")
        raise AssertionError("Test failed")


def test_openapi_docs():
    """Test OpenAPI documentation endpoints."""
    logger.info("=" * 80)
    logger.info("TEST 9: OpenAPI Documentation")
    logger.info("=" * 80)
    
    try:
        # Test Swagger UI
        response = requests.get(f"{API_BASE_URL}/docs")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        logger.info("✓ Swagger UI accessible at /docs")
        
        # Test ReDoc
        response = requests.get(f"{API_BASE_URL}/redoc")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        logger.info("✓ ReDoc accessible at /redoc")
        
        # Test OpenAPI schema
        response = requests.get(f"{API_BASE_URL}/openapi.json")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        logger.info(f"✓ OpenAPI {schema['openapi']} schema available")
        logger.info(f"✓ API Title: {schema['info']['title']}")
        logger.info(f"✓ API Version: {schema['info']['version']}")
        
        logger.info("✅ OpenAPI documentation test passed!\n")
        assert True
        
    except Exception as e:
        logger.error(f"❌ OpenAPI documentation test failed: {e}")
        raise AssertionError("Test failed")


def main():
    """Run all Phase 1.2 tests."""
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "PHASE 1.2: REST API FRAMEWORK TESTS" + " " * 23 + "║")
    logger.info("╚" + "═" * 78 + "╝\n")
    
    logger.info("⚠️  NOTE: Make sure the API server is running!")
    logger.info("   Start with: python -m sentinelzer0.api.server")
    logger.info("   Or: uvicorn sentinelzer0.api.server:app --reload\n")
    
    # Wait for user confirmation
    input("Press Enter when the server is ready...")
    
    results = []
    
    try:
        # Run all tests
        results.append(("Root Endpoint", test_root_endpoint()))
        results.append(("Health Check", test_health_endpoint()))
        results.append(("Authentication", test_authentication()))
        results.append(("Metrics Endpoint", test_metrics_endpoint()))
        results.append(("Model Info", test_model_info_endpoint()))
        results.append(("Single Prediction", test_predict_endpoint_single()))
        results.append(("Batch Prediction", test_predict_endpoint_batch()))
        results.append(("Configuration", test_config_endpoint()))
        results.append(("OpenAPI Docs", test_openapi_docs()))
        
        # Print summary
        logger.info("=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"  {status}: {test_name}")
        
        logger.info(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("\n🎉 ALL PHASE 1.2 TESTS PASSED!")
            logger.info("\n📊 Summary:")
            logger.info("  ✓ REST API endpoints operational")
            logger.info("  ✓ Authentication working")
            logger.info("  ✓ Real-time predictions functioning")
            logger.info("  ✓ Batch processing efficient")
            logger.info("  ✓ OpenAPI documentation available")
            logger.info("\n🎯 Phase 1.2 is complete!")
            return 0
        else:
            logger.error(f"\n❌ {total - passed} test(s) failed")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Test suite failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
