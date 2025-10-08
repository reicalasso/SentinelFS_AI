"""
SentinelZer0 Adversarial Robustness Module
==========================================

This module provides comprehensive adversarial robustness capabilities including:
- Attack generation (FGSM, PGD, C&W, Boundary attacks)
- Adversarial training
- Defense mechanisms (input sanitization, gradient masking, ensemble defenses)
- Robustness testing and evaluation
- Security validation

Version: 3.8.0
Phase: 4.1 - Adversarial Robustness
"""

from .attack_generator import (
    AttackGenerator,
    FGSMAttack,
    PGDAttack,
    CarliniWagnerAttack,
    BoundaryAttack
)

from .adversarial_trainer import (
    AdversarialTrainer,
    TrainingConfig
)

from .defense_mechanisms import (
    DefenseMechanism,
    InputSanitizer,
    GradientMasking,
    EnsembleDefense,
    AdversarialDetector
)

from .robustness_tester import (
    RobustnessTester,
    RobustnessMetrics
)

from .security_validator import (
    SecurityValidator,
    ValidationConfig
)

__version__ = "3.8.0"
__phase__ = "4.1"

__all__ = [
    # Attack Generation
    "AttackGenerator",
    "FGSMAttack",
    "PGDAttack",
    "CarliniWagnerAttack",
    "BoundaryAttack",
    
    # Adversarial Training
    "AdversarialTrainer",
    "TrainingConfig",
    
    # Defense Mechanisms
    "DefenseMechanism",
    "InputSanitizer",
    "GradientMasking",
    "EnsembleDefense",
    "AdversarialDetector",
    
    # Testing & Validation
    "RobustnessTester",
    "RobustnessMetrics",
    "SecurityValidator",
    "ValidationConfig",
]
