"""
Phase 4.1: Adversarial Robustness - Comprehensive Test Suite
============================================================

Tests all adversarial robustness components:
- Attack generation (FGSM, PGD, C&W, Boundary)
- Adversarial training
- Defense mechanisms
- Robustness testing
- Security validation
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import tempfile
import os

from sentinelzer0.adversarial_robustness import (
    AttackGenerator,
    AttackConfig,
    FGSMAttack,
    PGDAttack,
    CarliniWagnerAttack,
    BoundaryAttack,
    AdversarialTrainer,
    TrainingConfig,
    InputSanitizer,
    GradientMasking,
    EnsembleDefense,
    AdversarialDetector,
    DefenseConfig,
    RobustnessTester,
    RobustnessMetrics,
    SecurityValidator,
    ValidationConfig
)


# Simple test model
class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self, input_dim=10, hidden_dim=32, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@pytest.fixture
def device():
    """Get test device"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def model(device):
    """Create test model"""
    model = SimpleModel(input_dim=10, hidden_dim=32, num_classes=2)
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def sample_data(device):
    """Create sample data"""
    x = torch.randn(8, 10, device=device)
    y = torch.randint(0, 2, (8,), device=device)
    return x, y


@pytest.fixture
def data_loader(device):
    """Create test data loader"""
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=16, shuffle=True)


# ============================================================================
# ATTACK GENERATION TESTS
# ============================================================================

class TestAttackGeneration:
    """Test adversarial attack generation"""
    
    def test_fgsm_attack(self, model, sample_data, device):
        """Test FGSM attack generation"""
        x, y = sample_data
        config = AttackConfig(epsilon=0.3)
        attack = FGSMAttack(model, config)
        
        x_adv = attack.generate(x, y)
        
        assert x_adv.shape == x.shape
        assert not torch.equal(x_adv, x), "Adversarial examples should differ from originals"
        
        # Check perturbation is bounded
        perturbation = (x_adv - x).abs().max()
        assert perturbation <= config.epsilon * 1.1, "Perturbation exceeds epsilon"
    
    def test_pgd_attack(self, model, sample_data, device):
        """Test PGD attack generation"""
        x, y = sample_data
        config = AttackConfig(epsilon=0.3, num_steps=10)
        attack = PGDAttack(model, config)
        
        x_adv = attack.generate(x, y)
        
        assert x_adv.shape == x.shape
        assert not torch.equal(x_adv, x)
        
        # Check perturbation is bounded
        perturbation = torch.norm(x_adv - x, p=float('inf'), dim=1).max()
        assert perturbation <= config.epsilon * 1.1
    
    def test_cw_attack(self, model, sample_data, device):
        """Test Carlini & Wagner attack"""
        x, y = sample_data
        config = AttackConfig(
            num_steps=20,
            binary_search_steps=3,
            learning_rate=0.01
        )
        attack = CarliniWagnerAttack(model, config)
        
        x_adv = attack.generate(x[:4], y[:4])  # Use smaller batch
        
        assert x_adv.shape[0] == 4
        assert x_adv.shape[1:] == x.shape[1:]
    
    def test_boundary_attack(self, model, sample_data, device):
        """Test Boundary attack"""
        x, y = sample_data
        config = AttackConfig(num_steps=20)
        attack = BoundaryAttack(model, config)
        
        x_adv = attack.generate(x, y)
        
        assert x_adv.shape == x.shape
    
    def test_attack_generator(self, model, sample_data, device):
        """Test unified attack generator"""
        x, y = sample_data
        generator = AttackGenerator(model)
        
        # Test each attack type
        for attack_type in ['fgsm', 'pgd']:
            x_adv = generator.generate(x, y, attack_type)
            assert x_adv.shape == x.shape
            assert not torch.equal(x_adv, x)
    
    def test_evaluate_robustness(self, model, sample_data, device):
        """Test robustness evaluation"""
        x, y = sample_data
        generator = AttackGenerator(model)
        
        results = generator.evaluate_robustness(x, y, attack_types=['fgsm', 'pgd'])
        
        assert 'clean' in results
        assert 'fgsm' in results
        assert 'pgd' in results
        assert 'accuracy' in results['clean']
        assert 'accuracy' in results['fgsm']
        assert 'success_rate' in results['fgsm']


# ============================================================================
# ADVERSARIAL TRAINING TESTS
# ============================================================================

class TestAdversarialTraining:
    """Test adversarial training"""
    
    def test_training_config(self):
        """Test training configuration"""
        config = TrainingConfig(
            epochs=5,
            batch_size=32,
            learning_rate=0.001
        )
        
        assert config.epochs == 5
        assert config.batch_size == 32
        assert isinstance(config.attack_config, AttackConfig)
    
    def test_trainer_initialization(self, model, device):
        """Test trainer initialization"""
        config = TrainingConfig(epochs=2)
        trainer = AdversarialTrainer(model, config, device)
        
        assert trainer.model == model
        assert trainer.device == device
        assert trainer.attack_generator is not None
    
    def test_train_epoch(self, model, data_loader, device):
        """Test single epoch training"""
        config = TrainingConfig(epochs=1, warmup_epochs=0, adversarial_ratio=0.5)
        trainer = AdversarialTrainer(model, config, device)
        
        loss, acc = trainer.train_epoch(data_loader, epoch=1)
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 100
    
    def test_evaluate(self, model, data_loader, device):
        """Test model evaluation"""
        config = TrainingConfig()
        trainer = AdversarialTrainer(model, config, device)
        
        loss, acc = trainer.evaluate(data_loader, use_adversarial=False)
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 100
    
    def test_save_load_checkpoint(self, model, device):
        """Test checkpoint saving and loading"""
        config = TrainingConfig(epochs=1)
        trainer = AdversarialTrainer(model, config, device)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")
            trainer.save_checkpoint(checkpoint_path)
            
            assert os.path.exists(checkpoint_path)
            
            # Create new trainer and load
            new_model = SimpleModel().to(device)
            new_trainer = AdversarialTrainer(new_model, config, device)
            new_trainer.load_checkpoint(checkpoint_path)
            
            # Check states match
            assert new_trainer.best_acc == trainer.best_acc


# ============================================================================
# DEFENSE MECHANISMS TESTS
# ============================================================================

class TestDefenseMechanisms:
    """Test defense mechanisms"""
    
    def test_input_sanitizer(self, sample_data, device):
        """Test input sanitization"""
        x, _ = sample_data
        config = DefenseConfig(use_denoising=True, use_smoothing=True)
        sanitizer = InputSanitizer(config)
        
        x_clean = sanitizer.apply(x)
        
        assert x_clean.shape == x.shape
        # Should be different but similar
        assert not torch.equal(x_clean, x)
        assert torch.allclose(x_clean, x, atol=0.5)
    
    def test_feature_squeezing(self, sample_data, device):
        """Test feature squeezing"""
        x, _ = sample_data
        sanitizer = InputSanitizer()
        
        x_squeezed = sanitizer.feature_squeezing(x, bit_depth=4)
        
        assert x_squeezed.shape == x.shape
    
    def test_gradient_masking(self, model, sample_data, device):
        """Test gradient masking"""
        x, _ = sample_data
        config = DefenseConfig(use_gradient_masking=True, noise_scale=0.1)
        masking = GradientMasking(model, config)
        
        masking.training = True
        x_masked = masking.apply(x)
        
        assert x_masked.shape == x.shape
    
    def test_ensemble_defense(self, model, sample_data, device):
        """Test ensemble defense"""
        x, _ = sample_data
        
        # Create multiple models
        models = [SimpleModel().to(device) for _ in range(3)]
        config = DefenseConfig(use_ensemble=True)
        ensemble = EnsembleDefense(models, config)
        
        predictions = ensemble.predict_ensemble(x, method='average')
        
        assert predictions.shape[0] == x.shape[0]
        assert predictions.shape[1] == 2  # num_classes
    
    def test_ensemble_diversity(self, model, sample_data, device):
        """Test ensemble diversity calculation"""
        x, _ = sample_data
        
        models = [SimpleModel().to(device) for _ in range(3)]
        ensemble = EnsembleDefense(models)
        
        diversity = ensemble.calculate_diversity(x)
        
        assert isinstance(diversity, float)
        assert 0 <= diversity <= 1
    
    def test_adversarial_detector_calibration(self, model, data_loader, device):
        """Test adversarial detector calibration"""
        config = DefenseConfig(use_detection=True)
        detector = AdversarialDetector(model, config)
        
        detector.calibrate(data_loader)
        
        assert detector.clean_stats['mean_confidence'] is not None
        assert detector.clean_stats['mean_entropy'] is not None
    
    def test_adversarial_detection(self, model, sample_data, device):
        """Test adversarial example detection"""
        x, _ = sample_data
        config = DefenseConfig(use_detection=True, detection_threshold=0.5)
        detector = AdversarialDetector(model, config)
        
        detection_scores, is_adversarial = detector.detect(x, method='consistency')
        
        assert detection_scores.shape[0] == x.shape[0]
        assert is_adversarial.shape[0] == x.shape[0]
        assert is_adversarial.dtype == torch.bool


# ============================================================================
# ROBUSTNESS TESTING TESTS
# ============================================================================

class TestRobustnessTester:
    """Test robustness testing framework"""
    
    def test_robustness_metrics(self):
        """Test metrics container"""
        metrics = RobustnessMetrics()
        metrics.clean_accuracy = 0.95
        metrics.adversarial_accuracy = {'fgsm': 0.70, 'pgd': 0.60}
        metrics.robust_accuracy = 0.60
        
        # Test to_dict
        metrics_dict = metrics.to_dict()
        assert metrics_dict['clean_accuracy'] == 0.95
        assert 'fgsm' in metrics_dict['adversarial_accuracy']
        
        # Test summary
        summary = metrics.summary()
        assert 'ROBUSTNESS TEST SUMMARY' in summary
        assert 'Clean Accuracy' in summary
    
    def test_tester_initialization(self, model, device):
        """Test tester initialization"""
        tester = RobustnessTester(model, device=device)
        
        assert tester.model == model
        assert tester.device == device
        assert tester.attack_generator is not None
    
    def test_clean_accuracy(self, model, data_loader, device):
        """Test clean accuracy calculation"""
        tester = RobustnessTester(model, device=device)
        
        accuracy, confidence = tester.test_clean_accuracy(data_loader)
        
        assert isinstance(accuracy, float)
        assert isinstance(confidence, float)
        assert 0 <= accuracy <= 1
        assert 0 <= confidence <= 1
    
    def test_adversarial_attack_test(self, model, data_loader, device):
        """Test adversarial attack testing"""
        config = AttackConfig(epsilon=0.3, num_steps=10)
        tester = RobustnessTester(model, attack_config=config, device=device)
        
        results = tester.test_adversarial_attack(
            data_loader,
            attack_type='fgsm',
            use_defense=False
        )
        
        assert 'accuracy' in results
        assert 'confidence' in results
        assert 'l2_distance' in results
        assert 'linf_distance' in results
        assert 'time' in results
    
    def test_comprehensive_test(self, model, data_loader, device):
        """Test comprehensive robustness evaluation"""
        config = AttackConfig(epsilon=0.3, num_steps=5)
        tester = RobustnessTester(model, attack_config=config, device=device)
        
        metrics = tester.test_comprehensive(
            data_loader,
            attack_types=['fgsm'],
            use_defense=False
        )
        
        assert isinstance(metrics, RobustnessMetrics)
        assert metrics.clean_accuracy > 0
        assert 'fgsm' in metrics.adversarial_accuracy
        assert metrics.total_samples > 0
    
    def test_save_results(self, model, data_loader, device):
        """Test saving results"""
        tester = RobustnessTester(model, device=device)
        tester.metrics.clean_accuracy = 0.95
        tester.metrics.total_samples = 100
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "results.json")
            tester.save_results(filepath)
            
            assert os.path.exists(filepath)
            
            # Check can be loaded
            import json
            with open(filepath) as f:
                loaded = json.load(f)
            assert loaded['clean_accuracy'] == 0.95


# ============================================================================
# SECURITY VALIDATION TESTS
# ============================================================================

class TestSecurityValidator:
    """Test security validation"""
    
    def test_validation_config(self):
        """Test validation configuration"""
        config = ValidationConfig(
            window_size=500,
            validation_interval=50,
            enable_attack_detection=True
        )
        
        assert config.window_size == 500
        assert config.validation_interval == 50
        assert config.enable_attack_detection is True
    
    def test_validator_initialization(self, model, device):
        """Test validator initialization"""
        config = ValidationConfig()
        validator = SecurityValidator(model, config, device)
        
        assert validator.model == model
        assert validator.device == device
        assert validator.detector is not None
        assert len(validator.events) == 0
    
    def test_validate_sample(self, model, sample_data, device):
        """Test sample validation"""
        x, y = sample_data
        config = ValidationConfig(enable_attack_detection=True)
        validator = SecurityValidator(model, config, device)
        
        results = validator.validate_sample(x, y, return_details=True)
        
        assert 'is_safe' in results
        assert 'warnings' in results
        assert 'detections' in results
        assert 'statistics' in results
    
    def test_calibration(self, model, data_loader, device):
        """Test validator calibration"""
        validator = SecurityValidator(model, device=device)
        
        validator.calibrate(data_loader)
        
        assert validator.detector.clean_stats['mean_confidence'] is not None
    
    def test_get_statistics(self, model, sample_data, device):
        """Test statistics retrieval"""
        x, y = sample_data
        validator = SecurityValidator(model, device=device)
        
        # Process some samples
        validator.validate_sample(x, y)
        
        stats = validator.get_statistics()
        
        assert 'total_samples' in stats
        assert 'accuracy' in stats
        assert stats['total_samples'] > 0
    
    def test_security_events(self, model, sample_data, device):
        """Test security event logging"""
        x, y = sample_data
        config = ValidationConfig(enable_attack_detection=True)
        validator = SecurityValidator(model, config, device)
        
        # Process samples
        validator.validate_sample(x, y)
        
        events = validator.get_events()
        
        assert isinstance(events, list)
    
    def test_generate_security_report(self, model, sample_data, device):
        """Test security report generation"""
        x, y = sample_data
        validator = SecurityValidator(model, device=device)
        
        validator.validate_sample(x, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "security_report.md")
            validator.generate_security_report(filepath)
            
            assert os.path.exists(filepath)
            
            with open(filepath) as f:
                content = f.read()
            assert 'Security Validation Report' in content


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_attack_and_defense_integration(self, model, sample_data, device):
        """Test attack generation + defense application"""
        x, y = sample_data
        
        # Generate attack
        attack_config = AttackConfig(epsilon=0.3)
        generator = AttackGenerator(model, attack_config)
        x_adv = generator.generate(x, y, 'fgsm')
        
        # Apply defense
        defense_config = DefenseConfig(use_denoising=True)
        sanitizer = InputSanitizer(defense_config)
        x_defended = sanitizer.apply(x_adv)
        
        assert x_defended.shape == x.shape
    
    def test_full_pipeline(self, model, data_loader, device):
        """Test complete robustness testing pipeline"""
        # Setup
        attack_config = AttackConfig(epsilon=0.3, num_steps=5)
        defense_config = DefenseConfig(use_detection=True)
        
        # Test robustness
        tester = RobustnessTester(
            model,
            attack_config=attack_config,
            defense_config=defense_config,
            device=device
        )
        
        metrics = tester.test_comprehensive(
            data_loader,
            attack_types=['fgsm'],
            use_defense=True
        )
        
        assert metrics.clean_accuracy > 0
        assert metrics.total_samples > 0


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
