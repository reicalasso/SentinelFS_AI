"""
Comprehensive Tests for Phase 3.2: Explainability & Interpretability Framework
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from pathlib import Path
import tempfile

from sentinelzer0.explainability import (
    SHAPExplainer,
    LIMEExplainer,
    FeatureImportanceAnalyzer,
    DecisionReasoningEngine,
    AuditTrailSystem,
    ConfidenceScorer,
    ExplainabilityManager
)


class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=64, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def model():
    """Create test model."""
    model = DummyModel(input_dim=64, num_classes=2)
    model.eval()
    return model


@pytest.fixture
def sample_input():
    """Create sample input."""
    return torch.randn(1, 64)


@pytest.fixture
def batch_input():
    """Create batch input."""
    return torch.randn(8, 64)


@pytest.fixture
def feature_names():
    """Create feature names."""
    return [f"feature_{i}" for i in range(64)]


@pytest.fixture
def class_names():
    """Create class names."""
    return ['benign', 'malicious']


# ==================== SHAP Explainer Tests ====================

def test_shap_kernel_explainer(model, sample_input):
    """Test SHAP Kernel explainer."""
    explainer = SHAPExplainer(model)
    shap_values = explainer.explain(sample_input, method='kernel', n_samples=50)
    
    assert shap_values is not None
    assert hasattr(shap_values, 'values')
    assert hasattr(shap_values, 'base_value')
    print(f"✓ SHAP Kernel explainer: {shap_values.values.shape}")


def test_shap_top_features(model, sample_input, feature_names):
    """Test SHAP top features extraction."""
    explainer = SHAPExplainer(model)
    shap_values = explainer.explain(sample_input, method='kernel', n_samples=50)
    
    top_features = shap_values.top_features(k=5, feature_names=feature_names)
    
    assert len(top_features) == 5
    assert all(isinstance(name, str) for name, _ in top_features)
    print(f"✓ SHAP top features: {[name for name, _ in top_features[:3]]}")


# ==================== LIME Explainer Tests ====================

def test_lime_explainer(model, sample_input, feature_names):
    """Test LIME explainer."""
    explainer = LIMEExplainer(model, feature_names=feature_names)
    explanation = explainer.explain(sample_input, n_samples=100)
    
    assert explanation is not None
    assert hasattr(explanation, 'coefficients')
    assert hasattr(explanation, 'intercept')
    assert hasattr(explanation, 'score')
    print(f"✓ LIME explainer: score={explanation.score:.3f}")


def test_lime_batch_explain(model, batch_input, feature_names):
    """Test LIME batch explanation."""
    explainer = LIMEExplainer(model, feature_names=feature_names)
    explanations = explainer.explain_batch(batch_input, n_samples=50)
    
    assert len(explanations) == batch_input.shape[0]
    assert all(hasattr(exp, 'coefficients') for exp in explanations)
    print(f"✓ LIME batch: {len(explanations)} explanations")


# ==================== Feature Importance Tests ====================

def test_permutation_importance(model, sample_input, feature_names):
    """Test permutation importance."""
    analyzer = FeatureImportanceAnalyzer(model, feature_names=feature_names)
    importance = analyzer.compute_importance(sample_input, method='permutation', n_repeats=5)
    
    assert importance is not None
    assert hasattr(importance, 'scores')
    assert len(importance.scores) == len(feature_names)
    print(f"✓ Permutation importance: {len(importance.scores)} scores")


def test_gradient_importance(model, sample_input, feature_names):
    """Test gradient-based importance."""
    analyzer = FeatureImportanceAnalyzer(model, feature_names=feature_names)
    importance = analyzer.compute_importance(sample_input, method='gradient')
    
    assert importance is not None
    assert importance.method == 'gradient'
    print(f"✓ Gradient importance computed")


def test_compare_methods(model, sample_input, feature_names):
    """Test comparing multiple importance methods."""
    analyzer = FeatureImportanceAnalyzer(model, feature_names=feature_names)
    comparison = analyzer.compare_methods(sample_input, methods=['permutation', 'gradient'])
    
    assert 'permutation' in comparison
    assert 'gradient' in comparison
    assert 'consensus' in comparison
    print(f"✓ Method comparison: {list(comparison.keys())}")


# ==================== Decision Reasoning Tests ====================

def test_decision_reasoning(model, sample_input, feature_names, class_names):
    """Test decision reasoning engine."""
    engine = DecisionReasoningEngine(feature_names=feature_names, class_names=class_names)
    
    # Mock feature importance
    importance_scores = {name: np.random.rand() for name in feature_names[:10]}
    
    explanation = engine.explain_decision(sample_input, model, importance_scores)
    
    assert explanation is not None
    assert hasattr(explanation, 'explanation_text')
    assert hasattr(explanation, 'key_factors')
    assert hasattr(explanation, 'risk_level')
    print(f"✓ Decision reasoning: {explanation.risk_level}")


def test_counterfactual_generation(model, sample_input, feature_names, class_names):
    """Test counterfactual generation."""
    engine = DecisionReasoningEngine(feature_names=feature_names, class_names=class_names)
    
    counterfactual = engine.generate_counterfactual(sample_input, model)
    
    assert counterfactual is not None
    assert 'counterfactual_text' in counterfactual
    assert 'changes_needed' in counterfactual
    print(f"✓ Counterfactual: {len(counterfactual['changes_needed'])} changes")


# ==================== Audit Trail Tests ====================

def test_audit_trail_logging():
    """Test audit trail logging."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = Path(tmp.name)
    
    try:
        audit = AuditTrailSystem(db_path=db_path)
        
        # Log decision
        entry_id = audit.log_decision(
            input_data=[1.0] * 64,
            prediction=1,
            confidence=0.85,
            explanation={'method': 'test'},
            metadata={'test': True}
        )
        
        assert entry_id is not None
        print(f"✓ Audit trail logging: entry_id={entry_id}")
    
    finally:
        db_path.unlink(missing_ok=True)


def test_audit_trail_query():
    """Test audit trail querying."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = Path(tmp.name)
    
    try:
        audit = AuditTrailSystem(db_path=db_path)
        
        # Log multiple decisions
        for i in range(10):
            audit.log_decision(
                input_data=[float(i)] * 64,
                prediction=i % 2,
                confidence=0.5 + (i * 0.05),
                explanation={'idx': i}
            )
        
        # Query
        entries = audit.query_by_prediction(prediction=1)
        
        assert len(entries) == 5  # Half should be prediction=1
        print(f"✓ Audit trail query: {len(entries)} entries")
    
    finally:
        db_path.unlink(missing_ok=True)


def test_audit_trail_statistics():
    """Test audit trail statistics."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = Path(tmp.name)
    
    try:
        audit = AuditTrailSystem(db_path=db_path)
        
        # Log some decisions
        for _ in range(5):
            audit.log_decision(
                input_data=[1.0] * 64,
                prediction=1,
                confidence=0.9
            )
        
        stats = audit.get_statistics()
        
        assert stats['total_entries'] == 5
        assert 'avg_confidence' in stats
        print(f"✓ Audit statistics: {stats['total_entries']} entries")
    
    finally:
        db_path.unlink(missing_ok=True)


# ==================== Confidence Scorer Tests ====================

def test_confidence_scoring(model, sample_input):
    """Test confidence scoring."""
    scorer = ConfidenceScorer(model)
    score = scorer.score(sample_input)
    
    assert score is not None
    assert hasattr(score, 'raw_confidence')
    assert hasattr(score, 'calibrated_confidence')
    assert hasattr(score, 'uncertainty')
    assert 0 <= score.calibrated_confidence <= 1
    print(f"✓ Confidence score: {score.calibrated_confidence:.3f}")


def test_mc_dropout_confidence(model, sample_input):
    """Test MC dropout confidence."""
    scorer = ConfidenceScorer(model)
    score = scorer.score(sample_input, use_dropout=True, n_samples=10)
    
    assert score is not None
    assert score.method == 'mc_dropout'
    assert hasattr(score, 'uncertainty')
    print(f"✓ MC dropout confidence: uncertainty={score.uncertainty:.4f}")


def test_temperature_calibration(model, batch_input):
    """Test temperature calibration."""
    scorer = ConfidenceScorer(model)
    labels = torch.randint(0, 2, (batch_input.shape[0],))
    
    initial_temp = scorer.temperature
    scorer.calibrate_temperature(batch_input, labels, max_iter=10)
    
    assert scorer.temperature != initial_temp
    print(f"✓ Temperature calibration: {initial_temp:.2f} -> {scorer.temperature:.2f}")


def test_calibration_assessment(model, batch_input):
    """Test calibration assessment."""
    scorer = ConfidenceScorer(model)
    labels = torch.randint(0, 2, (batch_input.shape[0],))
    
    metrics = scorer.assess_calibration(batch_input, labels, n_bins=5)
    
    assert 'expected_calibration_error' in metrics
    assert 'bin_metrics' in metrics
    print(f"✓ Calibration assessment: ECE={metrics['expected_calibration_error']:.4f}")


# ==================== Explainability Manager Tests ====================

def test_explainability_manager_init(model, feature_names, class_names):
    """Test explainability manager initialization."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = Path(tmp.name)
    
    try:
        manager = ExplainabilityManager(
            model=model,
            feature_names=feature_names,
            class_names=class_names,
            audit_db_path=db_path
        )
        
        assert manager is not None
        assert hasattr(manager, 'shap_explainer')
        assert hasattr(manager, 'lime_explainer')
        assert hasattr(manager, 'confidence_scorer')
        print(f"✓ Explainability manager initialized")
    
    finally:
        db_path.unlink(missing_ok=True)


def test_comprehensive_explanation(model, sample_input, feature_names, class_names):
    """Test comprehensive explanation."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = Path(tmp.name)
    
    try:
        manager = ExplainabilityManager(
            model=model,
            feature_names=feature_names,
            class_names=class_names,
            audit_db_path=db_path
        )
        
        explanation = manager.explain(
            sample_input,
            methods=['confidence', 'reasoning'],
            log_to_audit=True
        )
        
        assert explanation is not None
        assert 'prediction' in explanation
        assert 'confidence' in explanation
        assert 'reasoning' in explanation
        print(f"✓ Comprehensive explanation: {explanation['prediction_label']}")
    
    finally:
        db_path.unlink(missing_ok=True)


def test_batch_explanation(model, batch_input, feature_names, class_names):
    """Test batch explanation."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = Path(tmp.name)
    
    try:
        manager = ExplainabilityManager(
            model=model,
            feature_names=feature_names,
            class_names=class_names,
            audit_db_path=db_path
        )
        
        explanations = manager.explain_batch(
            batch_input,
            methods=['confidence'],
            log_to_audit=False
        )
        
        assert len(explanations) == batch_input.shape[0]
        print(f"✓ Batch explanation: {len(explanations)} explanations")
    
    finally:
        db_path.unlink(missing_ok=True)


def test_report_generation(model, sample_input, feature_names, class_names):
    """Test report generation."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = Path(tmp.name)
    
    try:
        manager = ExplainabilityManager(
            model=model,
            feature_names=feature_names,
            class_names=class_names,
            audit_db_path=db_path
        )
        
        report = manager.generate_report(sample_input)
        
        assert report is not None
        assert 'COMPREHENSIVE EXPLANATION REPORT' in report
        assert 'CONFIDENCE ASSESSMENT' in report
        print(f"✓ Report generated: {len(report)} chars")
    
    finally:
        db_path.unlink(missing_ok=True)


def test_manager_statistics(model, feature_names, class_names):
    """Test manager statistics."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = Path(tmp.name)
    
    try:
        manager = ExplainabilityManager(
            model=model,
            feature_names=feature_names,
            class_names=class_names,
            audit_db_path=db_path
        )
        
        stats = manager.get_statistics()
        
        assert 'cache_size' in stats
        assert 'audit_stats' in stats
        assert 'components' in stats
        print(f"✓ Manager statistics: {list(stats.keys())}")
    
    finally:
        db_path.unlink(missing_ok=True)


# ==================== Integration Tests ====================

def test_end_to_end_explainability(model, sample_input, feature_names, class_names):
    """Test complete end-to-end explainability workflow."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = Path(tmp.name)
    
    try:
        # Initialize manager
        manager = ExplainabilityManager(
            model=model,
            feature_names=feature_names,
            class_names=class_names,
            audit_db_path=db_path
        )
        
        # Generate explanation
        explanation = manager.explain(
            sample_input,
            methods=['confidence', 'reasoning'],
            log_to_audit=True
        )
        
        # Generate report
        report = manager.generate_report(sample_input)
        
        # Check audit trail
        stats = manager.get_statistics()
        
        assert explanation is not None
        assert report is not None
        assert stats['audit_stats']['total_entries'] >= 1
        
        print(f"✓ End-to-end workflow: {stats['audit_stats']['total_entries']} audit entries")
    
    finally:
        db_path.unlink(missing_ok=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
