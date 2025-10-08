"""
Ensemble Manager - Unified Interface

Orchestrates all ensemble components for robust predictions.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import json

from .voting_system import EnsembleVoter, VotingStrategy, VotingResult
from .model_architectures import CNNDetector, LSTMDetector, TransformerDetector, DeepMLPDetector
from .training_pipeline import EnsembleTrainer, TrainingConfig
from .diversity_metrics import DiversityAnalyzer, DiversityMetrics


class EnsembleManager:
    """
    Unified interface for ensemble management.
    
    Coordinates:
    - Model training and loading
    - Ensemble predictions
    - Diversity analysis
    - Voting strategies
    - Performance monitoring
    """
    
    def __init__(
        self,
        models: Optional[List[nn.Module]] = None,
        voting_strategy: VotingStrategy = VotingStrategy.SOFT,
        device: Optional[torch.device] = None
    ):
        """
        Initialize ensemble manager.
        
        Args:
            models: List of pre-trained models
            voting_strategy: Strategy for combining predictions
            device: Device to run inference on
        """
        self.logger = logging.getLogger(__name__)
        self.models = models or []
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Components
        self.voter = EnsembleVoter(strategy=voting_strategy)
        self.diversity_analyzer = DiversityAnalyzer()
        self.trainer = None
        
        # Move models to device
        for model in self.models:
            model.to(self.device)
            model.eval()
        
        self.logger.info(
            f"Initialized ensemble manager with {len(self.models)} models "
            f"using {voting_strategy.value} voting"
        )
    
    def train_ensemble(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        config: Optional[TrainingConfig] = None,
        architectures: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train ensemble of models.
        
        Args:
            train_data: Training data
            train_labels: Training labels
            val_data: Validation data
            val_labels: Validation labels
            config: Training configuration
            architectures: List of architectures to use
        
        Returns:
            Training history
        """
        if config is None:
            config = TrainingConfig()
        
        # Create trainer
        self.trainer = EnsembleTrainer(config, device=self.device)
        
        # Create ensemble
        input_dim = train_data.shape[1]
        num_classes = len(torch.unique(train_labels))
        
        self.models = self.trainer.create_ensemble(
            input_dim=input_dim,
            num_classes=num_classes,
            architectures=architectures
        )
        
        # Train
        history = self.trainer.train(
            train_data, train_labels,
            val_data, val_labels
        )
        
        self.logger.info("Ensemble training complete")
        
        return history
    
    def predict(
        self,
        inputs: torch.Tensor,
        return_details: bool = False
    ) -> VotingResult:
        """
        Make ensemble prediction.
        
        Args:
            inputs: Input tensor
            return_details: Return detailed voting information
        
        Returns:
            Voting result
        """
        if len(self.models) == 0:
            raise ValueError("No models loaded")
        
        inputs = inputs.to(self.device)
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(inputs)
                predictions.append(output)
        
        # Vote
        result = self.voter.vote(predictions, return_details=return_details)
        
        return result
    
    def predict_batch(
        self,
        inputs: torch.Tensor,
        batch_size: int = 32
    ) -> List[VotingResult]:
        """
        Batch prediction.
        
        Args:
            inputs: Batch of inputs
            batch_size: Processing batch size
        
        Returns:
            List of voting results
        """
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            
            for j in range(len(batch)):
                single_input = batch[j:j+1]
                result = self.predict(single_input)
                results.append(result)
        
        return results
    
    def analyze_diversity(
        self,
        data: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> DiversityMetrics:
        """
        Analyze ensemble diversity.
        
        Args:
            data: Input data
            labels: True labels (optional)
        
        Returns:
            Diversity metrics
        """
        if len(self.models) == 0:
            raise ValueError("No models loaded")
        
        data = data.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        metrics = self.diversity_analyzer.compute_diversity(
            self.models, data, labels
        )
        
        return metrics
    
    def optimize_weights(
        self,
        val_data: torch.Tensor,
        val_labels: torch.Tensor
    ):
        """
        Optimize model weights for weighted voting.
        
        Args:
            val_data: Validation data
            val_labels: Validation labels
        """
        if len(self.models) == 0:
            raise ValueError("No models loaded")
        
        val_data = val_data.to(self.device)
        val_labels = val_labels.to(self.device)
        
        # Get predictions from all models
        val_predictions = []
        for model in self.models:
            model.eval()
            preds = []
            with torch.no_grad():
                for i in range(len(val_data)):
                    output = model(val_data[i:i+1])
                    preds.append(output)
            val_predictions.append(preds)
        
        # Compute optimal weights
        weights = self.voter.compute_optimal_weights(val_predictions, val_labels)
        
        # Update voter
        self.voter.set_weights(weights)
        self.voter.strategy = VotingStrategy.WEIGHTED
        
        self.logger.info(f"Optimized weights: {weights}")
    
    def evaluate(
        self,
        test_data: torch.Tensor,
        test_labels: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Evaluate ensemble performance.
        
        Args:
            test_data: Test data
            test_labels: Test labels
        
        Returns:
            Evaluation metrics
        """
        test_data = test_data.to(self.device)
        test_labels = test_labels.to(self.device)
        
        # Ensemble predictions
        ensemble_correct = 0
        ensemble_confidences = []
        individual_accuracies = [0] * len(self.models)
        
        for i in range(len(test_data)):
            inputs = test_data[i:i+1]
            label = test_labels[i].item()
            
            result = self.predict(inputs)
            
            if result.prediction == label:
                ensemble_correct += 1
            
            ensemble_confidences.append(result.confidence)
            
            # Individual model accuracies
            for j, pred in enumerate(result.individual_predictions):
                if pred == label:
                    individual_accuracies[j] += 1
        
        # Metrics
        ensemble_accuracy = ensemble_correct / len(test_data)
        individual_accuracies = [acc / len(test_data) for acc in individual_accuracies]
        avg_confidence = sum(ensemble_confidences) / len(ensemble_confidences)
        
        # Diversity
        diversity_metrics = self.analyze_diversity(test_data, test_labels)
        
        return {
            'ensemble_accuracy': float(ensemble_accuracy),
            'individual_accuracies': individual_accuracies,
            'avg_confidence': float(avg_confidence),
            'diversity_score': float(diversity_metrics.diversity_score),
            'disagreement': float(diversity_metrics.disagreement),
            'n_models': len(self.models)
        }
    
    def save(self, save_dir: Path):
        """
        Save ensemble.
        
        Args:
            save_dir: Directory to save ensemble
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for i, model in enumerate(self.models):
            model_path = save_dir / f"model_{i}.pt"
            torch.save({
                'state_dict': model.state_dict(),
                'architecture': model.__class__.__name__
            }, model_path)
            self.logger.info(f"Saved model {i} to {model_path}")
        
        # Save configuration
        config = {
            'n_models': len(self.models),
            'voting_strategy': self.voter.strategy.value,
            'weights': self.voter.weights,
            'model_architectures': [m.__class__.__name__ for m in self.models]
        }
        
        config_path = save_dir / "ensemble_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Saved ensemble configuration to {config_path}")
    
    def load(self, load_dir: Path, model_configs: Optional[List[Dict]] = None):
        """
        Load ensemble.
        
        Args:
            load_dir: Directory with saved ensemble
            model_configs: Model configurations (if needed)
        """
        load_dir = Path(load_dir)
        
        # Load configuration
        config_path = load_dir / "ensemble_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            n_models = config['n_models']
            voting_strategy = VotingStrategy(config['voting_strategy'])
            weights = config.get('weights')
            architectures = config.get('model_architectures', [])
        else:
            raise FileNotFoundError(f"Configuration not found at {config_path}")
        
        # Load models
        self.models = []
        for i in range(n_models):
            model_path = load_dir / f"model_{i}.pt"
            
            checkpoint = torch.load(model_path, map_location=self.device)
            arch_name = checkpoint.get('architecture', architectures[i] if i < len(architectures) else None)
            
            # Create model instance
            if arch_name == 'CNNDetector':
                model = CNNDetector()
            elif arch_name == 'LSTMDetector':
                model = LSTMDetector()
            elif arch_name == 'TransformerDetector':
                model = TransformerDetector()
            elif arch_name == 'DeepMLPDetector':
                model = DeepMLPDetector()
            else:
                raise ValueError(f"Unknown architecture: {arch_name}")
            
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(self.device)
            model.eval()
            
            self.models.append(model)
            self.logger.info(f"Loaded model {i} from {model_path}")
        
        # Update voter
        self.voter = EnsembleVoter(strategy=voting_strategy, weights=weights)
        
        self.logger.info(f"Loaded ensemble with {len(self.models)} models")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get ensemble statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'n_models': len(self.models),
            'voting_strategy': self.voter.strategy.value,
            'model_architectures': [m.__class__.__name__ for m in self.models],
            'weights': self.voter.weights,
            'device': str(self.device)
        }
    
    def visualize_performance(
        self,
        evaluation_results: Dict[str, Any]
    ) -> str:
        """
        Create text visualization of performance.
        
        Args:
            evaluation_results: Results from evaluate()
        
        Returns:
            Formatted string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("Ensemble Performance Report")
        lines.append("=" * 70)
        lines.append(f"Number of Models: {evaluation_results['n_models']}")
        lines.append(f"Voting Strategy: {self.voter.strategy.value}")
        lines.append("")
        lines.append("Accuracy Metrics:")
        lines.append("-" * 70)
        lines.append(f"  Ensemble Accuracy:  {evaluation_results['ensemble_accuracy']:.2%}")
        lines.append(f"  Average Confidence: {evaluation_results['avg_confidence']:.2%}")
        lines.append("")
        lines.append("Individual Model Accuracies:")
        lines.append("-" * 70)
        for i, acc in enumerate(evaluation_results['individual_accuracies']):
            arch = self.models[i].__class__.__name__
            lines.append(f"  Model {i} ({arch:20s}): {acc:.2%}")
        lines.append("")
        lines.append("Diversity Metrics:")
        lines.append("-" * 70)
        lines.append(f"  Diversity Score: {evaluation_results['diversity_score']:.4f}")
        lines.append(f"  Disagreement:    {evaluation_results['disagreement']:.4f}")
        lines.append("")
        
        # Performance assessment
        ensemble_acc = evaluation_results['ensemble_accuracy']
        best_individual = max(evaluation_results['individual_accuracies'])
        improvement = ensemble_acc - best_individual
        
        lines.append("Performance Assessment:")
        lines.append("-" * 70)
        if improvement > 0.01:
            lines.append(f"  ✅ Ensemble improves over best model by {improvement:.2%}")
        elif improvement > 0:
            lines.append(f"  ✓ Ensemble slightly better (+{improvement:.2%})")
        else:
            lines.append(f"  ⚠ Ensemble similar to best model ({improvement:+.2%})")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
