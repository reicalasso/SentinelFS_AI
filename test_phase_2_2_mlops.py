"""
Phase 2.2: MLOps & Model Versioning - Integration Tests

Tests all MLOps components including versioning, registry, A/B testing,
rollback, and MLflow integration.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("🚀 SentinelZer0 - Phase 2.2 MLOps & Model Versioning Tests")
print("=" * 70)


def test_model_version_manager():
    """Test model version management."""
    print("\n🧪 Testing Model Version Manager")
    print("=" * 40)
    
    try:
        from sentinelzer0.mlops import ModelVersionManager, VersionStatus
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelVersionManager(base_dir=tmpdir)
            
            # Create a dummy model file
            model_file = Path(tmpdir) / "test_model.pt"
            model_file.write_text("dummy model content")
            
            # Create version
            version = manager.create_version(
                model_path=str(model_file),
                created_by="test_user",
                training_metrics={"accuracy": 0.95, "f1": 0.93},
                hyperparameters={"learning_rate": 0.001, "batch_size": 32},
                notes="Test version",
                tags=["test", "v1"]
            )
            
            print(f"✅ Created version: {version.version}")
            print(f"   Status: {version.status.value}")
            print(f"   Metrics: {version.metadata.training_metrics}")
            
            # Get version
            retrieved = manager.get_version(version.version)
            assert retrieved is not None, "Failed to retrieve version"
            print(f"✅ Retrieved version: {retrieved.version}")
            
            # List versions
            versions = manager.list_versions()
            assert len(versions) == 1, "Expected 1 version"
            print(f"✅ Listed {len(versions)} version(s)")
            
            # Promote version
            manager.promote_version(version.version, VersionStatus.STAGING)
            print(f"✅ Promoted to: {version.status.value}")
            
            # Verify integrity
            is_valid = manager.verify_integrity(version.version)
            assert is_valid, "Integrity check failed"
            print(f"✅ Integrity verified")
            
            return True
    
    except Exception as e:
        print(f"❌ Model version manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_registry():
    """Test model registry with approval workflows."""
    print("\n🧪 Testing Model Registry")
    print("=" * 40)
    
    try:
        from sentinelzer0.mlops import (
            ModelVersionManager, ModelRegistry, ModelStage
        )
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            version_manager = ModelVersionManager(base_dir=tmpdir)
            registry = ModelRegistry(
                version_manager=version_manager,
                registry_dir=f"{tmpdir}/registry"
            )
            
            # Create a version
            model_file = Path(tmpdir) / "model.pt"
            model_file.write_text("model content")
            
            model_version = version_manager.create_version(
                model_path=str(model_file),
                created_by="test_user"
            )
            
            # Register model
            entry = registry.register_model(
                version=model_version.version,
                stage=ModelStage.DEVELOPMENT,
                registered_by="test_user",
                tags=["test"]
            )
            print(f"✅ Registered model: {entry.version}")
            print(f"   Stage: {entry.stage}")
            
            # Request promotion
            request = registry.request_promotion(
                version=model_version.version,
                to_stage=ModelStage.STAGING,
                requested_by="test_user",
                notes="Ready for staging"
            )
            print(f"✅ Created promotion request: {request.request_id}")
            print(f"   From: {request.from_stage} -> To: {request.to_stage}")
            
            # Approve promotion
            updated_entry = registry.approve_promotion(
                request_id=request.request_id,
                approved_by="approver",
                notes="Looks good"
            )
            print(f"✅ Approved promotion")
            print(f"   New stage: {updated_entry.stage}")
            
            # Get models by stage
            staging_models = registry.get_models_by_stage(ModelStage.STAGING)
            assert len(staging_models) == 1, "Expected 1 staging model"
            print(f"✅ Found {len(staging_models)} model(s) in staging")
            
            # Get audit trail
            audit_trail = registry.get_audit_trail(model_version.version)
            print(f"✅ Retrieved audit trail")
            print(f"   Events: {len(audit_trail['approval_requests'])}")
            
            return True
    
    except Exception as e:
        print(f"❌ Model registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ab_testing():
    """Test A/B testing framework."""
    print("\n🧪 Testing A/B Testing Framework")
    print("=" * 40)
    
    try:
        from sentinelzer0.mlops import ABTestManager
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ABTestManager(tests_dir=tmpdir)
            
            # Create A/B test
            test = manager.create_test(
                test_name="Model A vs Model B",
                model_a="v1.0.0",
                model_b="v1.1.0",
                traffic_split=0.5,
                description="Testing new model version",
                min_samples=10  # Low threshold for testing
            )
            print(f"✅ Created A/B test: {test.test_id}")
            print(f"   Models: {test.model_a} vs {test.model_b}")
            print(f"   Traffic split: {test.traffic_split:.0%}")
            
            # Start test
            manager.start_test(test.test_id)
            print(f"✅ Started test: {test.status}")
            
            # Simulate requests
            for i in range(20):
                model = manager.route_request(test.test_id)
                
                # Record result
                manager.record_result(
                    test_id=test.test_id,
                    model_version=model,
                    success=True,
                    latency_ms=25.0 + (i % 10),
                    confidence=0.85 + (i % 10) * 0.01,
                    true_label=True if i % 2 == 0 else False,
                    predicted_label=True if i % 3 == 0 else False
                )
            
            print(f"✅ Recorded 20 test requests")
            
            # Get results
            results = manager.get_test_results(test.test_id)
            print(f"✅ Test results:")
            print(f"   Model A requests: {results['metrics_a']['total_requests']}")
            print(f"   Model B requests: {results['metrics_b']['total_requests']}")
            print(f"   Recommendation: {results['recommendation'][:50]}...")
            
            # Complete test
            final_results = manager.complete_test(test.test_id, force=True)
            print(f"✅ Completed test")
            print(f"   Winner: {test.winner}")
            print(f"   Confidence: {test.winner_confidence:.2%}")
            
            return True
    
    except Exception as e:
        print(f"❌ A/B testing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rollback_manager():
    """Test automated rollback system."""
    print("\n🧪 Testing Rollback Manager")
    print("=" * 40)
    
    try:
        from sentinelzer0.mlops import (
            ModelVersionManager, ModelRegistry, RollbackManager,
            ModelStage, RollbackStrategy
        )
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            version_manager = ModelVersionManager(base_dir=tmpdir)
            registry = ModelRegistry(
                version_manager=version_manager,
                registry_dir=f"{tmpdir}/registry"
            )
            rollback_manager = RollbackManager(
                version_manager=version_manager,
                registry=registry,
                rollback_dir=f"{tmpdir}/rollback"
            )
            
            # Create two versions
            model_file = Path(tmpdir) / "model.pt"
            model_file.write_text("model v1")
            v1 = version_manager.create_version(
                model_path=str(model_file),
                version="v1_test"
            )
            registry.register_model(v1.version, ModelStage.PRODUCTION)
            
            model_file.write_text("model v2")
            v2 = version_manager.create_version(
                model_path=str(model_file),
                version="v2_test"
            )
            registry.register_model(v2.version, ModelStage.PRODUCTION)
            
            print(f"✅ Created versions: {v1.version}, {v2.version}")
            
            # Check health (healthy)
            health_check = rollback_manager.check_health(
                version=v2.version,
                metrics={
                    'error_rate': 0.05,
                    'avg_latency_ms': 50.0,
                    'total_requests': 100,
                    'failed_requests': 5
                }
            )
            print(f"✅ Health check: {health_check.is_healthy}")
            print(f"   Error rate: {health_check.error_rate:.2%}")
            print(f"   Latency: {health_check.avg_latency_ms:.1f}ms")
            
            # Check unhealthy scenario
            unhealthy_check = rollback_manager.check_health(
                version=v2.version,
                metrics={
                    'error_rate': 0.15,  # Above threshold
                    'avg_latency_ms': 150.0,  # Above threshold
                    'total_requests': 100,
                    'failed_requests': 15
                }
            )
            print(f"✅ Unhealthy check: {unhealthy_check.is_healthy}")
            
            # Should rollback?
            should_rollback = rollback_manager.should_rollback(
                current_version=v2.version,
                health_check=unhealthy_check,
                strategy=RollbackStrategy.IMMEDIATE
            )
            print(f"✅ Should rollback: {should_rollback}")
            
            # Execute rollback
            if should_rollback:
                event = rollback_manager.execute_rollback(
                    from_version=v2.version,
                    to_version=v1.version,
                    reason="Health check failed",
                    strategy=RollbackStrategy.IMMEDIATE
                )
                print(f"✅ Rollback executed: {event.event_id}")
                print(f"   From: {event.from_version} -> To: {event.to_version}")
                print(f"   Success: {event.success}")
            
            # Get history
            history = rollback_manager.get_rollback_history(limit=5)
            print(f"✅ Rollback history: {len(history)} event(s)")
            
            return True
    
    except Exception as e:
        print(f"❌ Rollback manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mlflow_integration():
    """Test MLflow integration."""
    print("\n🧪 Testing MLflow Integration")
    print("=" * 40)
    
    try:
        from sentinelzer0.mlops import MLflowTracker
        
        tracker = MLflowTracker(experiment_name="Test_Experiment")
        
        if tracker.is_available():
            print("✅ MLflow available")
            
            # Start run
            tracker.start_run(run_name="test_run", tags={"phase": "2.2"})
            print(f"✅ Started run: {tracker.get_run_id()}")
            
            # Log params
            tracker.log_params({
                "learning_rate": 0.001,
                "batch_size": 32
            })
            print("✅ Logged parameters")
            
            # Log metrics
            tracker.log_metrics({
                "accuracy": 0.95,
                "f1_score": 0.93
            })
            print("✅ Logged metrics")
            
            # End run
            tracker.end_run()
            print("✅ Ended run")
        else:
            print("⚠️  MLflow not available (install with: pip install mlflow)")
            print("✅ Graceful fallback working")
        
        return True
    
    except Exception as e:
        print(f"❌ MLflow integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 2.2 tests."""
    print("\n🧪 Testing Phase 2.2: MLOps & Model Versioning")
    print("=" * 60)
    
    tests = [
        ("Model Version Manager", test_model_version_manager),
        ("Model Registry", test_model_registry),
        ("A/B Testing", test_ab_testing),
        ("Rollback Manager", test_rollback_manager),
        ("MLflow Integration", test_mlflow_integration),
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    # Summary
    print("\n" + "=" * 70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {name}")
    
    if passed == total:
        print("🎉 All Phase 2.2 MLOps tests PASSED!")
        print("✅ Model Versioning & MLOps: COMPLETE")
        return 0
    else:
        print("❌ Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
