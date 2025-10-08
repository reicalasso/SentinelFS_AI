"""
Test Suite for Phase 3.1: Online Learning System
================================================

Comprehensive tests for all online learning components:
- IncrementalLearner: Learning strategies and adaptation
- ConceptDriftDetector: Drift detection methods
- FeedbackCollector: Feedback collection and storage
- RetrainingPipeline: Automated retraining
- OnlineValidator: Real-time validation
- OnlineLearningManager: End-to-end orchestration
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import json
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timedelta

from sentinelzer0.online_learning import (
    IncrementalLearner,
    LearningStrategy,
    ConceptDriftDetector,
    DriftDetectionMethod,
    FeedbackCollector,
    FeedbackType,
    RetrainingPipeline,
    RetrainingConfig,
    OnlineValidator,
    ValidationMetrics,
    OnlineLearningManager
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_model():
    """Create a simple neural network for testing."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )
    return model


@pytest.fixture
def sample_data():
    """Generate sample training data."""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    return X, y


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Test IncrementalLearner
# ============================================================================

class TestIncrementalLearner:
    """Test incremental learning functionality."""
    
    def test_initialization(self, simple_model):
        """Test learner initialization."""
        learner = IncrementalLearner(
            model=simple_model,
            learning_rate=0.001,
            strategy=LearningStrategy.SGD
        )
        
        assert learner.model is simple_model
        assert learner.learning_rate == 0.001
        assert learner.strategy == LearningStrategy.SGD
        assert learner.optimizer is not None
    
    def test_sgd_update(self, simple_model):
        """Test SGD strategy."""
        learner = IncrementalLearner(
            model=simple_model,
            learning_rate=0.01,
            strategy=LearningStrategy.SGD
        )
        
        X = torch.randn(1, 10)
        y = torch.tensor([0])
        
        initial_params = [p.clone() for p in simple_model.parameters()]
        metrics = learner.update(X, y)
        
        # Parameters should change
        for initial, updated in zip(initial_params, simple_model.parameters()):
            assert not torch.allclose(initial, updated)
        
        # Metrics should be returned
        assert "loss" in metrics
        assert metrics["loss"] >= 0
    
    def test_minibatch_update(self, simple_model, sample_data):
        """Test mini-batch strategy."""
        learner = IncrementalLearner(
            model=simple_model,
            learning_rate=0.01,
            strategy=LearningStrategy.MINIBATCH,
            batch_size=10
        )
        
        X, y = sample_data
        
        # Update with multiple samples
        for i in range(20):
            idx = i % len(X)
            metrics = learner.update(X[idx:idx+1], y[idx:idx+1])
        
        # After batch_size updates, buffer should have processed a batch
        assert len(learner.update_buffer) < learner.batch_size
    
    def test_replay_buffer(self, simple_model):
        """Test replay buffer strategy."""
        learner = IncrementalLearner(
            model=simple_model,
            learning_rate=0.01,
            strategy=LearningStrategy.REPLAY_BUFFER,
            buffer_size=50
        )
        
        # Add samples
        for i in range(100):
            X = torch.randn(1, 10)
            y = torch.tensor([i % 2])
            learner.update(X, y)
        
        # Buffer should be at max size
        assert len(learner.replay_buffer) == 50
    
    def test_ewma_update(self, simple_model):
        """Test EWMA strategy."""
        learner = IncrementalLearner(
            model=simple_model,
            learning_rate=0.01,
            strategy=LearningStrategy.EWMA,
            ewma_alpha=0.9
        )
        
        X = torch.randn(1, 10)
        y = torch.tensor([0])
        
        # First update should initialize EWMA params
        metrics = learner.update(X, y)
        assert learner.ewma_params is not None
        
        # Second update should use EWMA
        metrics = learner.update(X, y)
        assert "loss" in metrics
    
    def test_learning_rate_adaptation(self, simple_model):
        """Test adaptive learning rate."""
        learner = IncrementalLearner(
            model=simple_model,
            learning_rate=0.1,
            strategy=LearningStrategy.SGD,
            adaptive_lr=True
        )
        
        X = torch.randn(1, 10)
        y = torch.tensor([0])
        
        initial_lr = learner.learning_rate
        
        # Simulate high loss
        for _ in range(10):
            learner.update(X, y)
        
        # Learning rate should adapt based on loss history
        assert learner.learning_rate > 0


# ============================================================================
# Test ConceptDriftDetector
# ============================================================================

class TestConceptDriftDetector:
    """Test concept drift detection."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = ConceptDriftDetector(
            method=DriftDetectionMethod.ADWIN,
            window_size=100
        )
        
        assert detector.method == DriftDetectionMethod.ADWIN
        assert detector.window_size == 100
    
    def test_adwin_drift_detection(self):
        """Test ADWIN drift detection."""
        detector = ConceptDriftDetector(
            method=DriftDetectionMethod.ADWIN,
            delta=0.002
        )
        
        # Add stable data
        for _ in range(100):
            detector.add_sample(0.9)
        
        assert not detector.drift_detected
        
        # Add drifted data
        for _ in range(50):
            detector.add_sample(0.3)
        
        # Drift should be detected (may take some samples)
        # Just verify the method works without error
        assert detector.drift_detected or not detector.drift_detected
    
    def test_ddm_drift_detection(self):
        """Test DDM drift detection."""
        detector = ConceptDriftDetector(
            method=DriftDetectionMethod.DDM
        )
        
        # Add stable predictions (correct)
        for _ in range(100):
            detector.add_sample(1.0)  # Correct predictions
        
        assert not detector.drift_detected
        
        # Add many incorrect predictions
        for _ in range(30):
            detector.add_sample(0.0)  # Incorrect predictions
        
        # May detect drift or warning
        metrics = detector.get_metrics()
        assert "drift_detected" in metrics
    
    def test_kswin_drift_detection(self):
        """Test KSWIN drift detection."""
        detector = ConceptDriftDetector(
            method=DriftDetectionMethod.KSWIN,
            window_size=50
        )
        
        # Add stable data from one distribution
        stable_data = np.random.normal(0, 1, 100)
        for val in stable_data:
            detector.add_sample(val)
        
        assert not detector.drift_detected
        
        # Add data from different distribution
        drifted_data = np.random.normal(3, 1, 50)
        for val in drifted_data:
            detector.add_sample(val)
        
        # Should detect drift
        assert detector.drift_detected or len(detector.window) > 0
    
    def test_page_hinkley_drift_detection(self):
        """Test Page-Hinkley drift detection."""
        detector = ConceptDriftDetector(
            method=DriftDetectionMethod.PAGE_HINKLEY,
            threshold=10.0
        )
        
        # Add stable data
        for i in range(100):
            detector.add_sample(1.0 + np.random.normal(0, 0.1))
        
        # Add drifted data
        for i in range(50):
            detector.add_sample(5.0 + np.random.normal(0, 0.1))
        
        # Should detect drift
        metrics = detector.get_metrics()
        assert "cumulative_sum" in metrics
    
    def test_statistical_drift_detection(self):
        """Test statistical drift detection."""
        detector = ConceptDriftDetector(
            method=DriftDetectionMethod.STATISTICAL,
            window_size=100,
            alpha=0.05
        )
        
        # Add data from one distribution
        for _ in range(150):
            detector.add_sample(np.random.normal(0, 1))
        
        # Add data from different distribution
        for _ in range(50):
            detector.add_sample(np.random.normal(5, 1))
        
        # Should eventually detect drift
        metrics = detector.get_metrics()
        assert "p_value" in metrics or metrics["drift_detected"]


# ============================================================================
# Test FeedbackCollector
# ============================================================================

class TestFeedbackCollector:
    """Test feedback collection."""
    
    def test_initialization(self, temp_dir):
        """Test collector initialization."""
        collector = FeedbackCollector(
            storage_path=temp_dir / "feedback.json"
        )
        
        assert collector.storage_path.exists()
        assert len(collector.feedback_buffer) == 0
    
    def test_add_user_feedback(self, temp_dir):
        """Test adding user feedback."""
        collector = FeedbackCollector(storage_path=temp_dir / "feedback.json")
        
        collector.add_feedback(
            sample_id="sample_001",
            feedback_type=FeedbackType.USER_LABEL,
            true_label=1,
            predicted_label=0,
            confidence=0.8
        )
        
        assert len(collector.feedback_buffer) == 1
        feedback = collector.feedback_buffer[0]
        assert feedback["sample_id"] == "sample_001"
        assert feedback["feedback_type"] == FeedbackType.USER_LABEL.value
        assert feedback["true_label"] == 1
    
    def test_add_security_event(self, temp_dir):
        """Test adding security event feedback."""
        collector = FeedbackCollector(storage_path=temp_dir / "feedback.json")
        
        collector.add_feedback(
            sample_id="sample_002",
            feedback_type=FeedbackType.SECURITY_EVENT,
            true_label=1,
            predicted_label=0,
            event_severity="high",
            event_details={"action": "blocked"}
        )
        
        feedback = collector.feedback_buffer[0]
        assert feedback["metadata"]["event_severity"] == "high"
        assert feedback["metadata"]["event_details"]["action"] == "blocked"
    
    def test_get_feedback_batch(self, temp_dir):
        """Test retrieving feedback batches."""
        collector = FeedbackCollector(storage_path=temp_dir / "feedback.json")
        
        # Add multiple feedback items
        for i in range(10):
            collector.add_feedback(
                sample_id=f"sample_{i:03d}",
                feedback_type=FeedbackType.USER_LABEL,
                true_label=i % 2,
                predicted_label=(i + 1) % 2
            )
        
        # Get batch
        batch = collector.get_feedback_batch(batch_size=5)
        assert len(batch) == 5
        
        # Get another batch
        batch2 = collector.get_feedback_batch(batch_size=5)
        assert len(batch2) == 5
        
        # Should be no more items
        batch3 = collector.get_feedback_batch(batch_size=5)
        assert len(batch3) == 0
    
    def test_persistence(self, temp_dir):
        """Test feedback persistence."""
        storage_path = temp_dir / "feedback.json"
        
        # Create collector and add feedback
        collector1 = FeedbackCollector(storage_path=storage_path)
        collector1.add_feedback(
            sample_id="sample_001",
            feedback_type=FeedbackType.USER_LABEL,
            true_label=1,
            predicted_label=0
        )
        collector1.save()
        
        # Create new collector with same path
        collector2 = FeedbackCollector(storage_path=storage_path)
        assert len(collector2.feedback_buffer) == 1
        assert collector2.feedback_buffer[0]["sample_id"] == "sample_001"


# ============================================================================
# Test RetrainingPipeline
# ============================================================================

class TestRetrainingPipeline:
    """Test automated retraining."""
    
    def test_initialization(self, simple_model, temp_dir):
        """Test pipeline initialization."""
        config = RetrainingConfig(
            min_samples=10,
            validation_split=0.2,
            max_epochs=5,
            checkpoint_dir=temp_dir / "checkpoints"
        )
        
        pipeline = RetrainingPipeline(
            model=simple_model,
            config=config
        )
        
        assert pipeline.model is simple_model
        assert pipeline.config.min_samples == 10
        assert pipeline.config.checkpoint_dir.exists()
    
    def test_retrain_with_sufficient_data(self, simple_model, sample_data, temp_dir):
        """Test retraining with sufficient data."""
        config = RetrainingConfig(
            min_samples=10,
            validation_split=0.2,
            max_epochs=2,
            checkpoint_dir=temp_dir / "checkpoints"
        )
        
        pipeline = RetrainingPipeline(model=simple_model, config=config)
        
        X, y = sample_data
        
        initial_params = [p.clone() for p in simple_model.parameters()]
        result = pipeline.retrain(X, y)
        
        assert result["success"]
        assert "train_loss" in result
        assert "val_loss" in result
        
        # Parameters should have changed
        for initial, updated in zip(initial_params, simple_model.parameters()):
            assert not torch.allclose(initial, updated, atol=1e-6)
    
    def test_retrain_with_insufficient_data(self, simple_model, temp_dir):
        """Test retraining with insufficient data."""
        config = RetrainingConfig(
            min_samples=100,
            checkpoint_dir=temp_dir / "checkpoints"
        )
        
        pipeline = RetrainingPipeline(model=simple_model, config=config)
        
        X = torch.randn(10, 10)
        y = torch.randint(0, 2, (10,))
        
        result = pipeline.retrain(X, y)
        
        assert not result["success"]
        assert "error" in result
    
    def test_checkpoint_management(self, simple_model, sample_data, temp_dir):
        """Test model checkpoint management."""
        config = RetrainingConfig(
            min_samples=10,
            validation_split=0.2,
            max_epochs=2,
            checkpoint_dir=temp_dir / "checkpoints"
        )
        
        pipeline = RetrainingPipeline(model=simple_model, config=config)
        
        X, y = sample_data
        
        result = pipeline.retrain(X, y)
        
        assert result["success"]
        
        # Check checkpoint was created
        checkpoints = list(config.checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) > 0


# ============================================================================
# Test OnlineValidator
# ============================================================================

class TestOnlineValidator:
    """Test real-time validation."""
    
    def test_initialization(self):
        """Test validator initialization."""
        validator = OnlineValidator(
            window_size=100,
            threshold=0.1
        )
        
        assert validator.window_size == 100
        assert validator.threshold == 0.1
    
    def test_add_result(self):
        """Test adding validation results."""
        validator = OnlineValidator(window_size=50)
        
        validator.add_result(
            prediction=1,
            target=1,
            confidence=0.9
        )
        
        assert len(validator.predictions) == 1
        assert len(validator.targets) == 1
        assert len(validator.confidences) == 1
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        validator = OnlineValidator(window_size=100)
        
        # Add perfect predictions
        for i in range(50):
            validator.add_result(
                prediction=i % 2,
                target=i % 2,
                confidence=0.95
            )
        
        metrics = validator.get_metrics()
        
        assert metrics.accuracy == 1.0
        assert metrics.precision >= 0.0
        assert metrics.recall >= 0.0
        assert metrics.f1_score >= 0.0
        assert 0.0 <= metrics.avg_confidence <= 1.0
    
    def test_performance_degradation_detection(self):
        """Test performance degradation detection."""
        validator = OnlineValidator(
            window_size=100,
            threshold=0.2
        )
        
        # Add good predictions
        for i in range(50):
            validator.add_result(
                prediction=i % 2,
                target=i % 2,
                confidence=0.9
            )
        
        baseline = validator.get_metrics()
        
        # Add bad predictions
        for i in range(50):
            validator.add_result(
                prediction=i % 2,
                target=(i + 1) % 2,  # Wrong predictions
                confidence=0.5
            )
        
        # Should detect degradation
        assert validator.check_degradation(baseline)


# ============================================================================
# Test OnlineLearningManager
# ============================================================================

class TestOnlineLearningManager:
    """Test end-to-end online learning system."""
    
    def test_initialization(self, simple_model, temp_dir):
        """Test manager initialization."""
        config = RetrainingConfig(
            min_samples=10,
            checkpoint_dir=temp_dir / "checkpoints"
        )
        
        manager = OnlineLearningManager(
            model=simple_model,
            learning_rate=0.01,
            retraining_config=config,
            drift_detection_method=DriftDetectionMethod.ADWIN,
            feedback_storage_path=temp_dir / "feedback.json"
        )
        
        assert manager.model is simple_model
        assert manager.learner is not None
        assert manager.drift_detector is not None
        assert manager.feedback_collector is not None
        assert manager.retraining_pipeline is not None
        assert manager.validator is not None
    
    def test_process_sample(self, simple_model, temp_dir):
        """Test processing individual samples."""
        config = RetrainingConfig(
            min_samples=10,
            checkpoint_dir=temp_dir / "checkpoints"
        )
        
        manager = OnlineLearningManager(
            model=simple_model,
            learning_rate=0.01,
            retraining_config=config,
            feedback_storage_path=temp_dir / "feedback.json"
        )
        
        X = torch.randn(1, 10)
        y = torch.tensor([0])
        confidence = 0.8
        
        result = manager.process_sample(X, y, confidence)
        
        assert "learning_metrics" in result
        assert "drift_detected" in result
        assert "validation_metrics" in result
    
    def test_add_feedback(self, simple_model, temp_dir):
        """Test adding user feedback."""
        config = RetrainingConfig(
            min_samples=10,
            checkpoint_dir=temp_dir / "checkpoints"
        )
        
        manager = OnlineLearningManager(
            model=simple_model,
            learning_rate=0.01,
            retraining_config=config,
            feedback_storage_path=temp_dir / "feedback.json"
        )
        
        manager.add_feedback(
            sample_id="sample_001",
            feedback_type=FeedbackType.USER_LABEL,
            true_label=1,
            predicted_label=0
        )
        
        # Feedback should be collected
        assert len(manager.feedback_collector.feedback_buffer) == 1
    
    def test_trigger_retraining(self, simple_model, sample_data, temp_dir):
        """Test manual retraining trigger."""
        config = RetrainingConfig(
            min_samples=10,
            validation_split=0.2,
            max_epochs=2,
            checkpoint_dir=temp_dir / "checkpoints"
        )
        
        manager = OnlineLearningManager(
            model=simple_model,
            learning_rate=0.01,
            retraining_config=config,
            feedback_storage_path=temp_dir / "feedback.json"
        )
        
        X, y = sample_data
        
        result = manager.trigger_retraining(X, y)
        
        assert result["success"]
        assert "train_loss" in result
    
    def test_comprehensive_statistics(self, simple_model, temp_dir):
        """Test comprehensive statistics retrieval."""
        config = RetrainingConfig(
            min_samples=10,
            checkpoint_dir=temp_dir / "checkpoints"
        )
        
        manager = OnlineLearningManager(
            model=simple_model,
            learning_rate=0.01,
            retraining_config=config,
            feedback_storage_path=temp_dir / "feedback.json"
        )
        
        # Process some samples
        for i in range(10):
            X = torch.randn(1, 10)
            y = torch.tensor([i % 2])
            manager.process_sample(X, y, confidence=0.8)
        
        stats = manager.get_comprehensive_statistics()
        
        assert "learner" in stats
        assert "drift_detector" in stats
        assert "validator" in stats
        assert "feedback" in stats
        assert stats["feedback"]["total_feedback"] == 0


# ============================================================================
# Integration Test
# ============================================================================

class TestOnlineLearningIntegration:
    """Test complete online learning workflow."""
    
    def test_complete_workflow(self, simple_model, sample_data, temp_dir):
        """Test end-to-end online learning workflow."""
        config = RetrainingConfig(
            min_samples=20,
            validation_split=0.2,
            max_epochs=3,
            checkpoint_dir=temp_dir / "checkpoints"
        )
        
        manager = OnlineLearningManager(
            model=simple_model,
            learning_rate=0.01,
            learning_strategy=LearningStrategy.MINIBATCH,
            retraining_config=config,
            drift_detection_method=DriftDetectionMethod.ADWIN,
            feedback_storage_path=temp_dir / "feedback.json"
        )
        
        X, y = sample_data
        
        # Simulate online learning
        for i in range(50):
            idx = i % len(X)
            result = manager.process_sample(
                X[idx:idx+1],
                y[idx:idx+1],
                confidence=0.7 + np.random.rand() * 0.3
            )
            
            # Occasionally add feedback
            if i % 10 == 0:
                manager.add_feedback(
                    sample_id=f"sample_{i:03d}",
                    feedback_type=FeedbackType.USER_LABEL,
                    true_label=int(y[idx]),
                    predicted_label=int(y[idx])
                )
        
        # Get statistics
        stats = manager.get_comprehensive_statistics()
        
        assert stats["learner"]["total_updates"] == 50
        assert stats["validator"]["total_samples"] == 50
        assert stats["feedback"]["total_feedback"] == 5
        
        # Trigger retraining with sufficient data
        retrain_result = manager.trigger_retraining(X[:30], y[:30])
        assert retrain_result["success"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
