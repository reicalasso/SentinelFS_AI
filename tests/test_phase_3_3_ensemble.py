"""
Comprehensive Tests for Phase 3.3: Ensemble Management Framework
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from pathlib import Path
import tempfile

from sentinelzer0.ensemble import (
    EnsembleVoter, VotingStrategy, VotingResult,
    CNNDetector, LSTMDetector, TransformerDetector, DeepMLPDetector,
    EnsembleTrainer, TrainingConfig,
    DiversityAnalyzer, DiversityMetrics,
    EnsembleManager
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    torch.manual_seed(42)
    data = torch.randn(100, 64)
    labels = torch.randint(0, 2, (100,))
    return data, labels


@pytest.fixture
def small_data():
    """Create small dataset for quick tests."""
    torch.manual_seed(42)
    data = torch.randn(20, 64)
    labels = torch.randint(0, 2, (20,))
    return data, labels


@pytest.fixture
def test_models():
    """Create test models."""
    models = [
        CNNDetector(input_dim=64, num_classes=2),
        LSTMDetector(input_dim=64, num_classes=2),
        TransformerDetector(input_dim=64, num_classes=2)
    ]
    for model in models:
        model.eval()
    return models


# ==================== Voting System Tests ====================

def test_soft_voting(test_models):
    """Test soft voting."""
    voter = EnsembleVoter(strategy=VotingStrategy.SOFT)
    
    # Create mock predictions
    predictions = [
        torch.randn(1, 2) for _ in test_models
    ]
    
    result = voter.vote(predictions)
    
    assert isinstance(result, VotingResult)
    assert result.strategy == "soft"
    assert 0 <= result.confidence <= 1
    assert len(result.individual_predictions) == len(test_models)
    print(f"✓ Soft voting: confidence={result.confidence:.3f}")


def test_hard_voting(test_models):
    """Test hard voting."""
    voter = EnsembleVoter(strategy=VotingStrategy.HARD)
    
    predictions = [
        torch.randn(1, 2) for _ in test_models
    ]
    
    result = voter.vote(predictions)
    
    assert isinstance(result, VotingResult)
    assert result.strategy == "hard"
    assert result.prediction in [0, 1]
    print(f"✓ Hard voting: prediction={result.prediction}")


def test_weighted_voting(test_models):
    """Test weighted voting."""
    weights = [0.5, 0.3, 0.2]
    voter = EnsembleVoter(strategy=VotingStrategy.WEIGHTED, weights=weights)
    
    predictions = [
        torch.randn(1, 2) for _ in test_models
    ]
    
    result = voter.vote(predictions)
    
    assert isinstance(result, VotingResult)
    assert result.strategy == "weighted"
    assert result.weights is not None
    print(f"✓ Weighted voting: weights={result.weights}")


def test_uncertainty_computation(test_models):
    """Test uncertainty computation."""
    voter = EnsembleVoter(strategy=VotingStrategy.SOFT)
    
    predictions = [
        torch.randn(1, 2) for _ in test_models
    ]
    
    uncertainty = voter.get_uncertainty(predictions)
    
    assert 0 <= uncertainty <= 1
    print(f"✓ Uncertainty: {uncertainty:.4f}")


def test_agreement_computation(test_models):
    """Test agreement computation."""
    voter = EnsembleVoter(strategy=VotingStrategy.SOFT)
    
    predictions = [
        torch.randn(1, 2) for _ in test_models
    ]
    
    agreement = voter.get_agreement(predictions)
    
    assert 0 <= agreement <= 1
    print(f"✓ Agreement: {agreement:.4f}")


# ==================== Model Architecture Tests ====================

def test_cnn_detector():
    """Test CNN detector."""
    model = CNNDetector(input_dim=64, num_classes=2)
    x = torch.randn(4, 64)
    
    output = model(x)
    
    assert output.shape == (4, 2)
    print(f"✓ CNN detector: output shape={output.shape}")


def test_lstm_detector():
    """Test LSTM detector."""
    model = LSTMDetector(input_dim=64, num_classes=2)
    x = torch.randn(4, 64)
    
    output = model(x)
    
    assert output.shape == (4, 2)
    print(f"✓ LSTM detector: output shape={output.shape}")


def test_transformer_detector():
    """Test Transformer detector."""
    model = TransformerDetector(input_dim=64, num_classes=2)
    x = torch.randn(4, 64)
    
    output = model(x)
    
    assert output.shape == (4, 2)
    print(f"✓ Transformer detector: output shape={output.shape}")


def test_deep_mlp_detector():
    """Test Deep MLP detector."""
    model = DeepMLPDetector(input_dim=64, num_classes=2)
    x = torch.randn(4, 64)
    
    output = model(x)
    
    assert output.shape == (4, 2)
    print(f"✓ Deep MLP detector: output shape={output.shape}")


# ==================== Training Pipeline Tests ====================

def test_ensemble_creation(small_data):
    """Test ensemble creation."""
    config = TrainingConfig(epochs=1, batch_size=8)
    trainer = EnsembleTrainer(config)
    
    data, labels = small_data
    
    models = trainer.create_ensemble(
        input_dim=64,
        num_classes=2,
        architectures=['cnn', 'lstm']
    )
    
    assert len(models) == 2
    print(f"✓ Ensemble created: {len(models)} models")


def test_ensemble_training(small_data):
    """Test ensemble training."""
    config = TrainingConfig(epochs=2, batch_size=8, use_bagging=False)
    trainer = EnsembleTrainer(config)
    
    data, labels = small_data
    
    trainer.create_ensemble(
        input_dim=64,
        num_classes=2,
        architectures=['cnn', 'lstm']
    )
    
    history = trainer.train(data, labels)
    
    assert 'model_0' in history
    assert 'model_1' in history
    assert len(history['model_0']) == 2  # 2 epochs
    print(f"✓ Ensemble training: {len(history)} models trained")


# ==================== Diversity Metrics Tests ====================

def test_diversity_computation(test_models, small_data):
    """Test diversity computation."""
    analyzer = DiversityAnalyzer()
    data, labels = small_data
    
    metrics = analyzer.compute_diversity(test_models, data, labels)
    
    assert isinstance(metrics, DiversityMetrics)
    assert 0 <= metrics.diversity_score <= 1
    print(f"✓ Diversity score: {metrics.diversity_score:.4f}")


def test_disagreement_metric(test_models, small_data):
    """Test disagreement metric."""
    analyzer = DiversityAnalyzer()
    data, labels = small_data
    
    metrics = analyzer.compute_diversity(test_models, data, labels)
    
    assert 0 <= metrics.disagreement <= 1
    print(f"✓ Disagreement: {metrics.disagreement:.4f}")


def test_pairwise_analysis(test_models, small_data):
    """Test pairwise diversity analysis."""
    analyzer = DiversityAnalyzer()
    data, labels = small_data
    
    pairwise = analyzer.analyze_pairwise(test_models, data, labels)
    
    assert len(pairwise) == 3  # C(3,2) = 3 pairs
    assert (0, 1) in pairwise
    print(f"✓ Pairwise analysis: {len(pairwise)} pairs")


def test_diversity_visualization(test_models, small_data):
    """Test diversity visualization."""
    analyzer = DiversityAnalyzer()
    data, labels = small_data
    
    metrics = analyzer.compute_diversity(test_models, data, labels)
    viz = analyzer.visualize_diversity(metrics)
    
    assert isinstance(viz, str)
    assert "Diversity Score" in viz
    print(f"✓ Diversity visualization: {len(viz)} chars")


# ==================== Ensemble Manager Tests ====================

def test_manager_initialization():
    """Test ensemble manager initialization."""
    manager = EnsembleManager()
    
    assert manager is not None
    assert len(manager.models) == 0
    print(f"✓ Manager initialized")


def test_manager_with_models(test_models):
    """Test manager with pre-loaded models."""
    manager = EnsembleManager(models=test_models)
    
    assert len(manager.models) == len(test_models)
    print(f"✓ Manager with {len(test_models)} models")


def test_manager_prediction(test_models):
    """Test manager prediction."""
    manager = EnsembleManager(models=test_models)
    
    x = torch.randn(1, 64)
    result = manager.predict(x)
    
    assert isinstance(result, VotingResult)
    assert result.prediction in [0, 1]
    print(f"✓ Manager prediction: {result.prediction}, conf={result.confidence:.3f}")


def test_manager_batch_prediction(test_models):
    """Test manager batch prediction."""
    manager = EnsembleManager(models=test_models)
    
    x = torch.randn(10, 64)
    results = manager.predict_batch(x, batch_size=4)
    
    assert len(results) == 10
    assert all(isinstance(r, VotingResult) for r in results)
    print(f"✓ Batch prediction: {len(results)} results")


def test_manager_diversity_analysis(test_models, small_data):
    """Test manager diversity analysis."""
    manager = EnsembleManager(models=test_models)
    data, labels = small_data
    
    metrics = manager.analyze_diversity(data, labels)
    
    assert isinstance(metrics, DiversityMetrics)
    print(f"✓ Manager diversity: score={metrics.diversity_score:.4f}")


def test_manager_evaluation(test_models, small_data):
    """Test manager evaluation."""
    manager = EnsembleManager(models=test_models)
    data, labels = small_data
    
    results = manager.evaluate(data, labels)
    
    assert 'ensemble_accuracy' in results
    assert 'individual_accuracies' in results
    assert 'diversity_score' in results
    print(f"✓ Manager evaluation: acc={results['ensemble_accuracy']:.2%}")


def test_manager_save_load(test_models):
    """Test manager save and load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        
        # Save
        manager = EnsembleManager(models=test_models)
        manager.save(save_dir)
        
        # Load
        manager2 = EnsembleManager()
        manager2.load(save_dir)
        
        assert len(manager2.models) == len(test_models)
        print(f"✓ Manager save/load: {len(manager2.models)} models")


def test_manager_statistics(test_models):
    """Test manager statistics."""
    manager = EnsembleManager(models=test_models)
    
    stats = manager.get_statistics()
    
    assert 'n_models' in stats
    assert 'voting_strategy' in stats
    assert 'model_architectures' in stats
    print(f"✓ Manager statistics: {stats['n_models']} models")


def test_manager_performance_visualization(test_models, small_data):
    """Test manager performance visualization."""
    manager = EnsembleManager(models=test_models)
    data, labels = small_data
    
    eval_results = manager.evaluate(data, labels)
    viz = manager.visualize_performance(eval_results)
    
    assert isinstance(viz, str)
    assert "Ensemble Performance" in viz
    print(f"✓ Performance visualization: {len(viz)} chars")


# ==================== Integration Tests ====================

def test_end_to_end_ensemble(small_data):
    """Test complete ensemble workflow."""
    data, labels = small_data
    
    # Split data
    split = int(len(data) * 0.7)
    train_data, test_data = data[:split], data[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    
    # Create and train ensemble
    manager = EnsembleManager()
    
    config = TrainingConfig(epochs=2, batch_size=4, use_bagging=False)
    history = manager.train_ensemble(
        train_data, train_labels,
        architectures=['cnn', 'lstm']
    )
    
    # Evaluate
    results = manager.evaluate(test_data, test_labels)
    
    assert 'ensemble_accuracy' in results
    assert len(manager.models) == 2
    print(f"✓ End-to-end: acc={results['ensemble_accuracy']:.2%}, diversity={results['diversity_score']:.4f}")


def test_ensemble_with_optimization(small_data):
    """Test ensemble with weight optimization."""
    data, labels = small_data
    
    # Create models
    models = [
        CNNDetector(input_dim=64, num_classes=2),
        LSTMDetector(input_dim=64, num_classes=2)
    ]
    for model in models:
        model.eval()
    
    # Create manager
    manager = EnsembleManager(models=models, voting_strategy=VotingStrategy.SOFT)
    
    # Optimize weights
    manager.optimize_weights(data, labels)
    
    assert manager.voter.strategy == VotingStrategy.WEIGHTED
    assert manager.voter.weights is not None
    print(f"✓ Weight optimization: weights={manager.voter.weights}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
