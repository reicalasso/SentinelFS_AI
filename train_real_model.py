"""
Real-world SentinelFS AI System - Complete Integration Example

This script demonstrates:
1. Training a hybrid threat detection model from real data
2. Deploying the model for real-time inference
3. Monitoring and evaluation in production
4. Incremental learning and model updates
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random
import json

# Import our real-world components
from sentinelfs_ai.models.hybrid_detector import HybridThreatDetector
from sentinelfs_ai.data.real_feature_extractor import RealFeatureExtractor
from sentinelfs_ai.training.real_trainer import RealWorldTrainer
from sentinelfs_ai.inference.real_engine import RealTimeInferenceEngine
from sentinelfs_ai.evaluation.production_evaluator import ProductionEvaluator
from sentinelfs_ai.utils.logger import get_logger

logger = get_logger(__name__)


class SentinelFSAISystem:
    """
    Complete SentinelFS AI System for real-world deployment.
    """
    
    def __init__(
        self,
        model_dir: str = './models/production',
        metrics_dir: str = './metrics',
        checkpoint_dir: str = './checkpoints'
    ):
        self.model_dir = Path(model_dir)
        self.metrics_dir = Path(metrics_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.feature_extractor = RealFeatureExtractor(
            window_size=100,
            time_window_seconds=300
        )
        
        # Model will be initialized during training or loading
        self.model = None
        self.trainer = None
        self.inference_engine = None
        self.evaluator = ProductionEvaluator(
            metrics_dir=str(self.metrics_dir)
        )
        
        logger.info("SentinelFS AI System initialized")
    
    def train_from_real_data(
        self,
        train_events: list,
        val_events: list,
        train_labels: np.ndarray,
        val_labels: np.ndarray,
        hyperparameters: dict = None
    ):
        """
        Train the model from real file system events.
        
        Args:
            train_events: List of training event dictionaries
            val_events: List of validation event dictionaries
            train_labels: Training labels (0=normal, 1=anomaly)
            val_labels: Validation labels
            hyperparameters: Model and training hyperparameters
        """
        logger.info("="*80)
        logger.info("TRAINING HYBRID THREAT DETECTION MODEL")
        logger.info("="*80)
        
        # Default hyperparameters
        hp = hyperparameters or {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 50,
            'sequence_length': 50
        }
        
        # Initialize model
        num_features = self.feature_extractor.get_num_features()
        logger.info(f"Initializing model with {num_features} input features")
        
        self.model = HybridThreatDetector(
            input_size=num_features,
            hidden_size=hp['hidden_size'],
            num_layers=hp['num_layers'],
            dropout=hp['dropout'],
            use_gru=True,  # GRU is faster than LSTM
            isolation_forest_contamination=0.1,
            heuristic_weight=0.3,
            dl_weight=0.4,
            anomaly_weight=0.3
        )
        
        # Initialize trainer
        self.trainer = RealWorldTrainer(
            model=self.model,
            feature_extractor=self.feature_extractor,
            learning_rate=hp['learning_rate'],
            checkpoint_dir=str(self.checkpoint_dir)
        )
        
        # Train model
        training_results = self.trainer.train_from_real_data(
            train_events=train_events,
            val_events=val_events,
            train_labels=train_labels,
            val_labels=val_labels,
            num_epochs=hp['num_epochs'],
            batch_size=hp['batch_size'],
            sequence_length=hp['sequence_length']
        )
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETED")
        logger.info(f"Best validation loss: {training_results['best_val_loss']:.4f}")
        logger.info(f"Training time: {training_results['training_time']:.2f}s")
        logger.info(f"Final metrics: {training_results['final_metrics']}")
        logger.info("="*80)
        
        # Save model
        self.save_model()
        
        return training_results
    
    def deploy_for_inference(
        self,
        sequence_length: int = 50,
        threat_threshold: float = 0.5
    ):
        """
        Deploy the trained model for real-time inference.
        
        Args:
            sequence_length: Sequence length for analysis
            threat_threshold: Threshold for threat detection
        """
        logger.info("="*80)
        logger.info("DEPLOYING MODEL FOR REAL-TIME INFERENCE")
        logger.info("="*80)
        
        if self.model is None:
            logger.error("No model available. Train or load a model first.")
            return
        
        self.inference_engine = RealTimeInferenceEngine(
            model=self.model,
            feature_extractor=self.feature_extractor,
            sequence_length=sequence_length,
            threat_threshold=threat_threshold,
            enable_caching=True,
            enable_batching=True
        )
        
        logger.info("Model deployed successfully")
        logger.info(f"Threat threshold: {threat_threshold}")
        logger.info(f"Target latency: <25ms")
        logger.info("="*80)
    
    def analyze_event(self, event: dict, ground_truth: int = None) -> dict:
        """
        Analyze a single file system event.
        
        Args:
            event: Event dictionary
            ground_truth: Optional ground truth label for evaluation
            
        Returns:
            Analysis result dictionary
        """
        if self.inference_engine is None:
            raise RuntimeError("Model not deployed. Call deploy_for_inference() first.")
        
        # Perform inference
        result = self.inference_engine.analyze_event(
            event, 
            return_explanation=True
        )
        
        # Record for evaluation if ground truth provided
        if ground_truth is not None:
            self.evaluator.record_prediction(
                prediction=result.threat_score,
                ground_truth=ground_truth,
                metadata=event
            )
        
        return result.to_dict()
    
    def incremental_update(
        self,
        new_events: list,
        new_labels: np.ndarray
    ):
        """
        Perform incremental learning with new data.
        
        Args:
            new_events: New event data
            new_labels: Labels for new data
        """
        logger.info("="*80)
        logger.info("PERFORMING INCREMENTAL MODEL UPDATE")
        logger.info(f"New samples: {len(new_events)}")
        logger.info("="*80)
        
        if self.trainer is None:
            logger.error("Trainer not initialized")
            return
        
        update_results = self.trainer.incremental_update(
            new_events=new_events,
            new_labels=new_labels,
            num_epochs=5,
            batch_size=32
        )
        
        logger.info(f"Update completed - Loss: {update_results['avg_loss']:.4f}")
        logger.info("="*80)
        
        # Redeploy with updated model
        self.deploy_for_inference()
    
    def generate_evaluation_report(self) -> dict:
        """Generate comprehensive evaluation report."""
        logger.info("="*80)
        logger.info("GENERATING EVALUATION REPORT")
        logger.info("="*80)
        
        report = self.evaluator.generate_report()
        
        # Log key metrics
        metrics = report['performance_metrics']
        logger.info(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"Recall: {metrics.get('recall', 0):.4f}")
        logger.info(f"F1 Score: {metrics.get('f1_score', 0):.4f}")
        logger.info(f"False Positive Rate: {metrics.get('false_positive_rate', 0):.4f}")
        logger.info(f"False Negative Rate: {metrics.get('false_negative_rate', 0):.4f}")
        
        if 'latency_p99_ms' in metrics:
            logger.info(f"P99 Latency: {metrics['latency_p99_ms']:.2f}ms")
        
        # Log alerts
        if report['active_alerts']:
            logger.warning(f"Active alerts: {len(report['active_alerts'])}")
            for alert in report['active_alerts']:
                logger.warning(f"  - {alert['metric']}: {alert['value']:.4f} (threshold: {alert['threshold']})")
        
        # Log drift
        drift_info = report.get('drift_detection', {})
        if drift_info.get('drift_detected', False):
            logger.warning("MODEL DRIFT DETECTED!")
            logger.warning("Recommendation: Perform incremental update or retrain model")
        
        logger.info("="*80)
        
        return report
    
    def save_model(self, model_name: str = 'sentinelfs_production'):
        """Save the complete model."""
        logger.info(f"Saving model to {self.model_dir / model_name}")
        
        # Save PyTorch model
        model_path = self.model_dir / f"{model_name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'use_gru': self.model.use_gru
            },
            'feature_extractor_config': {
                'num_features': self.feature_extractor.get_num_features(),
                'feature_names': self.feature_extractor.get_feature_names()
            },
            'timestamp': datetime.now().isoformat()
        }, model_path)
        
        # Save additional components
        self.model.save_components(str(self.model_dir / model_name))
        
        logger.info("Model saved successfully")
    
    def load_model(self, model_name: str = 'sentinelfs_production'):
        """Load a saved model."""
        logger.info(f"Loading model from {self.model_dir / model_name}")
        
        model_path = self.model_dir / f"{model_name}.pt"
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return
        
        # Load checkpoint
        checkpoint = torch.load(model_path)
        config = checkpoint['model_config']
        
        # Recreate model
        self.model = HybridThreatDetector(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            use_gru=config['use_gru']
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load additional components
        self.model.load_components(str(self.model_dir / model_name))
        
        logger.info("Model loaded successfully")
    
    def get_performance_stats(self) -> dict:
        """Get current system performance statistics."""
        stats = {}
        
        if self.inference_engine:
            stats['inference'] = self.inference_engine.get_performance_stats()
        
        stats['evaluation'] = self.evaluator.calculate_metrics()
        
        return stats


def simulate_real_file_events(num_normal: int = 1000, num_anomaly: int = 200) -> tuple:
    """
    Simulate realistic file system events for demonstration.
    
    Returns:
        (events, labels)
    """
    events = []
    labels = []
    
    users = ['user1', 'user2', 'user3', 'admin']
    normal_extensions = ['.txt', '.pdf', '.doc', '.jpg', '.py', '.json']
    anomaly_extensions = ['.encrypted', '.locked', '.exe']
    
    base_time = datetime.now() - timedelta(days=7)
    
    # Generate normal events
    for i in range(num_normal):
        user = random.choice(users)
        ext = random.choice(normal_extensions)
        
        event = {
            'timestamp': (base_time + timedelta(minutes=i)).isoformat(),
            'user_id': user,
            'operation': random.choice(['read', 'write']),
            'file_path': f'/home/{user}/documents/file_{i}{ext}',
            'file_size': random.randint(1024, 10*1024*1024),  # 1KB to 10MB
        }
        events.append(event)
        labels.append(0)  # Normal
    
    # Generate anomalous events (ransomware-like)
    for i in range(num_anomaly):
        user = random.choice(users)
        
        # Simulate rapid encryption
        if random.random() > 0.5:
            # Ransomware pattern: rapid modifications with suspicious extensions
            event = {
                'timestamp': (base_time + timedelta(days=5, seconds=i)).isoformat(),
                'user_id': user,
                'operation': 'write',
                'file_path': f'/home/{user}/documents/important_{i}{random.choice(anomaly_extensions)}',
                'file_size': random.randint(1024, 5*1024*1024),
            }
        else:
            # Mass deletion or unusual time access
            event = {
                'timestamp': (base_time + timedelta(days=6, hours=2, seconds=i)).isoformat(),  # 2 AM
                'user_id': user,
                'operation': random.choice(['delete', 'rename']),
                'file_path': f'/home/{user}/documents/file_{i}.txt',
                'file_size': random.randint(1024, 1024*1024),
            }
        
        events.append(event)
        labels.append(1)  # Anomaly
    
    # Shuffle
    combined = list(zip(events, labels))
    random.shuffle(combined)
    events, labels = zip(*combined)
    
    return list(events), np.array(labels)


def main():
    """Main demonstration function."""
    
    print("\n" + "="*80)
    print("SENTINELFS AI - REAL-WORLD THREAT DETECTION SYSTEM")
    print("="*80 + "\n")
    
    # Initialize system
    system = SentinelFSAISystem()
    
    # Step 1: Generate simulated data
    print("Step 1: Generating simulated file system events...")
    all_events, all_labels = simulate_real_file_events(num_normal=2000, num_anomaly=400)
    
    # Split into train and val
    split_idx = int(len(all_events) * 0.8)
    train_events = all_events[:split_idx]
    train_labels = all_labels[:split_idx]
    val_events = all_events[split_idx:]
    val_labels = all_labels[split_idx:]
    
    print(f"Generated {len(train_events)} training events, {len(val_events)} validation events")
    print(f"Anomaly rate: {np.mean(all_labels)*100:.2f}%\n")
    
    # Step 2: Train model
    print("Step 2: Training hybrid threat detection model...")
    training_results = system.train_from_real_data(
        train_events=train_events,
        val_events=val_events,
        train_labels=train_labels,
        val_labels=val_labels,
        hyperparameters={
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 20,  # Reduced for demo
            'sequence_length': 50
        }
    )
    
    # Step 3: Deploy for inference
    print("\nStep 3: Deploying model for real-time inference...")
    system.deploy_for_inference(threat_threshold=0.5)
    
    # Step 4: Test inference
    print("\nStep 4: Testing real-time threat detection...")
    test_events = val_events[:100]
    test_labels = val_labels[:100]
    
    for event, label in zip(test_events, test_labels):
        result = system.analyze_event(event, ground_truth=label)
        
        if result['anomaly_detected']:
            print(f"  [THREAT] {event['file_path']} - Score: {result['threat_score']:.3f}")
    
    # Step 5: Generate evaluation report
    print("\nStep 5: Generating evaluation report...")
    report = system.generate_evaluation_report()
    
    # Step 6: Export metrics
    print("\nStep 6: Exporting performance metrics...")
    stats = system.get_performance_stats()
    
    print("\nInference Performance:")
    if 'inference' in stats:
        inf_stats = stats['inference']
        print(f"  Total inferences: {inf_stats['total_inferences']}")
        print(f"  Avg latency: {inf_stats['avg_latency_ms']:.2f}ms")
        print(f"  P95 latency: {inf_stats['p95_latency_ms']:.2f}ms")
        print(f"  P99 latency: {inf_stats['p99_latency_ms']:.2f}ms")
        print(f"  Threats detected: {inf_stats['threats_detected']}")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Integrate with SentinelFS Rust FUSE layer")
    print("2. Deploy in production with Prometheus monitoring")
    print("3. Set up incremental learning pipeline")
    print("4. Configure alerting thresholds")
    print("\nModel ready for production deployment!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
