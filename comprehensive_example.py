"""
Comprehensive example showcasing all enhanced features of SentinelFS AI.

This example demonstrates:
- Data generation with realistic patterns
- Model training with advanced techniques
- Model management and versioning
- Inference with explainability
- Evaluation with advanced metrics
- Ensemble methods
- Adversarial training and robustness
"""

import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Import all enhanced SentinelFS AI components
from sentinelfs_ai import (
    # Core types
    AnalysisResult, AnomalyType, TrainingConfig,
    
    # Models
    BehavioralAnalyzer, TransformerBehavioralAnalyzer, 
    CNNLSTMAnalyzer, EnsembleAnalyzer, AdaptiveAnalyzer,
    
    # Data processing
    FeatureExtractor, DataProcessor, generate_sample_data,
    generate_realistic_access_data,
    
    # Training
    train_model, train_robust, train_with_augmentation,
    calculate_metrics, evaluate_model, EarlyStopping,
    AdversarialTrainer, RobustnessEvaluator,
    
    # Evaluation
    AdvancedEvaluator, plot_roc_curve, plot_precision_recall_curve,
    plot_confusion_matrix, calibration_plot,
    
    # Inference
    InferenceEngine,
    
    # Management
    ModelManager, save_checkpoint, load_checkpoint,
    
    # Utils
    get_logger
)

def run_comprehensive_example():
    """Run the comprehensive example demonstrating all features."""
    logger = get_logger(__name__)
    logger.info("Starting comprehensive SentinelFS AI example...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Generate training data with realistic patterns
    logger.info("1. Generating realistic training data...")
    X, y, anomaly_types = generate_realistic_access_data(
        num_samples=2000,
        seq_len=20,
        anomaly_ratio=0.2,  # 20% anomalies
        complexity_level='high'  # More complex patterns
    )
    
    logger.info(f"Generated {len(X)} samples with shape {X.shape}")
    logger.info(f"Anomaly distribution: {np.bincount(y.flatten())}")
    logger.info(f"Anomaly types distribution: {np.bincount(anomaly_types.flatten()) if anomaly_types is not None else 'N/A'}")
    
    # 2. Set up data processing
    logger.info("2. Setting up data processing...")
    data_processor = DataProcessor(
        batch_size=64,
        val_split=0.2,
        test_split=0.1,
        scaler_type='standard',
        normalize_features=True
    )
    
    # Prepare data with anomaly types for multi-task learning
    data_loaders, multi_loaders = data_processor.prepare_data_with_anomaly_types(X, y, anomaly_types)
    
    # Get data statistics
    stats = data_processor.get_data_statistics(X, y)
    logger.info(f"Dataset statistics: {stats['label_distribution']}")
    
    # 3. Train multiple model architectures
    logger.info("3. Training multiple model architectures...")
    
    # 3a. Train Behavioral Analyzer (LSTM-based)
    logger.info("Training Behavioral Analyzer...")
    model_config = {
        'input_size': X.shape[2],  # Number of features
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.3,
        'use_attention': True,
        'bidirectional': True
    }
    
    lstm_model = BehavioralAnalyzer(**model_config)
    lstm_history = train_model(
        model=lstm_model,
        dataloaders=data_loaders,
        epochs=20,
        lr=0.001,
        patience=5,
        gradient_clipping=1.0,
        loss_function='bce',
        log_interval=5
    )
    
    # 3b. Train Transformer model
    logger.info("Training Transformer model...")
    transformer_model = TransformerBehavioralAnalyzer(
        input_size=X.shape[2],
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.3,
        seq_len=20
    )
    
    transformer_history = train_model(
        model=transformer_model,
        dataloaders=data_loaders,
        epochs=15,
        lr=0.0005,
        patience=5,
        gradient_clipping=1.0
    )
    
    # 3c. Train CNN-LSTM model
    logger.info("Training CNN-LSTM model...")
    cnn_lstm_model = CNNLSTMAnalyzer(
        input_size=X.shape[2],
        hidden_size=128,
        num_layers=3,
        dropout=0.3
    )
    
    cnn_lstm_history = train_model(
        model=cnn_lstm_model,
        dataloaders=data_loaders,
        epochs=15,
        lr=0.001,
        patience=5,
        gradient_clipping=1.0
    )
    
    # 4. Train with adversarial techniques for robustness
    logger.info("4. Training with adversarial techniques...")
    robust_model = BehavioralAnalyzer(**model_config)
    robust_history = train_robust(
        model=robust_model,
        dataloaders=data_loaders,
        epochs=15,
        lr=0.001,
        adversarial_training=True,
        adversarial_ratio=0.3,
        epsilon=0.01
    )
    
    # 5. Evaluate all models comprehensively
    logger.info("5. Evaluating models with advanced metrics...")
    evaluator = AdvancedEvaluator()
    
    # Evaluate LSTM model
    lstm_metrics = evaluator.evaluate_model_comprehensive(
        model=lstm_model,
        test_loader=data_loaders['test']
    )
    logger.info(f"LSTM Model - Accuracy: {lstm_metrics['accuracy']:.4f}, F1: {lstm_metrics['f1_score']:.4f}")
    
    # Evaluate Transformer model
    transformer_metrics = evaluator.evaluate_model_comprehensive(
        model=transformer_model,
        test_loader=data_loaders['test']
    )
    logger.info(f"Transformer Model - Accuracy: {transformer_metrics['accuracy']:.4f}, F1: {transformer_metrics['f1_score']:.4f}")
    
    # Evaluate CNN-LSTM model
    cnn_lstm_metrics = evaluator.evaluate_model_comprehensive(
        model=cnn_lstm_model,
        test_loader=data_loaders['test']
    )
    logger.info(f"CNN-LSTM Model - Accuracy: {cnn_lstm_metrics['accuracy']:.4f}, F1: {cnn_lstm_metrics['f1_score']:.4f}")
    
    # Evaluate robust model
    robust_metrics = evaluator.evaluate_model_comprehensive(
        model=robust_model,
        test_loader=data_loaders['test']
    )
    logger.info(f"Robust Model - Accuracy: {robust_metrics['accuracy']:.4f}, F1: {robust_metrics['f1_score']:.4f}")
    
    # 6. Test adversarial robustness
    logger.info("6. Testing adversarial robustness...")
    robustness_evaluator = RobustnessEvaluator(robust_model)
    robustness_results = robustness_evaluator.evaluate_robustness(
        torch.FloatTensor(X[:100]),  # Use subset for speed
        torch.FloatTensor(y[:100])
    )
    logger.info(f"Robustness - Clean accuracy: {robustness_results['clean_accuracy'][0]:.4f}")
    logger.info(f"Robustness - Adversarial accuracy: {robustness_results['adversarial_accuracy'][-1]:.4f}")
    
    # 7. Create and evaluate ensemble
    logger.info("7. Creating and evaluating ensemble model...")
    from sentinelfs_ai.training.ensemble_training import EnsembleManager
    
    ensemble_manager = EnsembleManager(
        input_size=X.shape[2],
        ensemble_size=3,
        base_architecture='mixed',  # Use different architectures
        hidden_size=128,
        num_layers=3
    )
    
    # Add trained models to ensemble (simplified approach)
    ensemble_models = [lstm_model, transformer_model, cnn_lstm_model]
    # Note: In a real scenario, we would train the ensemble properly
    # For this example, we'll create an ensemble by averaging predictions
    
    # Make ensemble predictions
    with torch.no_grad():
        X_test = torch.FloatTensor(X[-100:])  # Use last 100 samples as test
        y_test = torch.FloatTensor(y[-100:])
        
        # Get predictions from each model
        lstm_preds = lstm_model(X_test).cpu().numpy()
        transformer_preds = transformer_model(X_test).cpu().numpy()
        cnn_lstm_preds = cnn_lstm_model(X_test).cpu().numpy()
        
        # Ensemble prediction (average)
        ensemble_preds = (lstm_preds + transformer_preds + cnn_lstm_preds) / 3
        ensemble_accuracy = ((ensemble_preds > 0.5) == y_test.numpy()).mean()
        
        logger.info(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
    
    # 8. Model management and versioning
    logger.info("8. Managing models with versioning...")
    model_manager = ModelManager(model_dir=Path('./models_comprehensive'))
    
    # Save models with versioning
    feature_extractor = data_processor.feature_extractor
    
    model_manager.save_model(
        model=lstm_model,
        version='1.0.0',
        metrics=lstm_metrics,
        feature_extractor=feature_extractor,
        export_formats=['onnx', 'torchscript', 'quantized'],
        model_name='behavioral_analyzer'
    )
    
    model_manager.save_model(
        model=robust_model,
        version='1.1.0',
        metrics=robust_metrics,
        feature_extractor=feature_extractor,
        export_formats=['onnx', 'torchscript'],
        model_name='robust_analyzer'
    )
    
    # List available models
    available_models = model_manager.list_available_models()
    logger.info(f"Available models: {len(available_models)}")
    
    # 9. Load and use model for inference
    logger.info("9. Loading and using model for inference...")
    loaded_model, loaded_feature_extractor = model_manager.load_model(version='1.0.0')
    
    # Create inference engine
    inference_engine = InferenceEngine(
        model=loaded_model,
        feature_extractor=loaded_feature_extractor,
        threshold=0.5,
        enable_explainability=True,
        enable_performance_monitoring=True
    )
    
    # Test inference on a random sample
    random_sample = X[np.random.randint(0, len(X))]
    result = inference_engine.analyze(random_sample)
    
    logger.info(f"Inference result:")
    logger.info(f"  - Anomaly detected: {result.anomaly_detected}")
    logger.info(f"  - Confidence: {result.confidence:.4f}")
    logger.info(f"  - Threat score: {result.threat_score:.2f}")
    if result.anomaly_type:
        logger.info(f"  - Anomaly type: {result.anomaly_type}")
    
    # Get detailed explanation
    explanation = inference_engine.explain_prediction(random_sample, method='gradient')
    logger.info(f"  - Explanation summary: {', '.join(explanation.get('summary', []))}")
    
    # 10. Performance monitoring
    performance_metrics = inference_engine.get_performance_metrics()
    logger.info(f"Performance metrics: {performance_metrics}")
    
    # 11. Cross-validation
    logger.info("10. Performing cross-validation...")
    cv_results = evaluator.cross_validate(
        model_class=BehavioralAnalyzer,
        model_params=model_config,
        data=X[:500],  # Use subset for faster CV
        labels=y[:500],
        n_folds=3
    )
    
    avg_cv_accuracy = np.mean(cv_results['accuracy'])
    logger.info(f"Cross-validation average accuracy: {avg_cv_accuracy:.4f}")
    
    # 12. Stratified evaluation
    logger.info("11. Performing stratified evaluation...")
    stratified_results = evaluator.stratified_evaluation(
        model=loaded_model,
        data=X,
        labels=y,
        anomaly_types=anomaly_types
    )
    
    for stratum, metrics in stratified_results.items():
        logger.info(f"  {stratum}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}")
    
    # 13. Feature importance and statistics
    logger.info("12. Analyzing feature importance...")
    feature_stats = loaded_feature_extractor.get_feature_statistics(X)
    logger.info(f"Feature statistics: {list(feature_stats.keys())}")
    
    # 14. Advanced evaluation metrics
    logger.info("13. Calculating advanced metrics...")
    y_true = y.flatten()
    with torch.no_grad():
        y_pred_proba = lstm_model(torch.FloatTensor(X)).cpu().numpy().flatten()
    
    advanced_metrics = evaluator.calculate_comprehensive_metrics(y_true, y_pred_proba)
    logger.info(f"Advanced metrics - AUC-ROC: {advanced_metrics['auc_roc']:.4f}, "
                f"AUC-PR: {advanced_metrics['auc_pr']:.4f}, "
                f"Calibration error: {advanced_metrics['calibration_error']:.4f}")
    
    # 15. Model validation
    logger.info("14. Validating model integrity...")
    validation_result = model_manager.validate_model(
        loaded_model,
        torch.FloatTensor(X[:10]),
        (10, 1)
    )
    logger.info(f"Model validation result: {validation_result}")
    
    logger.info("Comprehensive example completed successfully!")
    
    # Summary of results
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'datasets': {
            'total_samples': len(X),
            'seq_length': X.shape[1],
            'num_features': X.shape[2],
            'anomaly_ratio': float(y.mean())
        },
        'model_performance': {
            'lstm': {'accuracy': lstm_metrics['accuracy'], 'f1': lstm_metrics['f1_score']},
            'transformer': {'accuracy': transformer_metrics['accuracy'], 'f1': transformer_metrics['f1_score']},
            'cnn_lstm': {'accuracy': cnn_lstm_metrics['accuracy'], 'f1': cnn_lstm_metrics['f1_score']},
            'robust': {'accuracy': robust_metrics['accuracy'], 'f1': robust_metrics['f1_score']},
            'ensemble': {'accuracy': float(ensemble_accuracy)}
        },
        'robustness': {
            'clean_accuracy': robustness_results['clean_accuracy'][0],
            'adversarial_accuracy': robustness_results['adversarial_accuracy'][-1],
            'epsilon': robustness_results['epsilon'][-1]
        },
        'cross_validation': {
            'avg_accuracy': float(avg_cv_accuracy),
            'std_accuracy': float(np.std(cv_results['accuracy']))
        }
    }
    
    logger.info(f"Results summary: {json.dumps(results_summary, indent=2, default=str)}")
    
    return results_summary


if __name__ == "__main__":
    run_comprehensive_example()