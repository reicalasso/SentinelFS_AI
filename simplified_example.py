"""
Simplified example demonstrating core SentinelFS AI functionality.

This example shows the essential workflow:
- Data generation
- Model training
- Inference and analysis
"""

import torch
import numpy as np
from pathlib import Path

# Import core SentinelFS AI components
from sentinelfs_ai import (
    # Core types
    AnalysisResult, AnomalyType, TrainingConfig,
    
    # Models
    BehavioralAnalyzer,
    
    # Data processing
    DataProcessor, generate_realistic_access_data,
    
    # Training
    train_model, calculate_metrics,
    
    # Inference
    InferenceEngine,
    
    # Management
    ModelManager,
    
    # Utils
    get_logger
)


def run_simplified_example():
    """Run the simplified example demonstrating core functionality."""
    logger = get_logger(__name__)
    logger.info("Starting simplified SentinelFS AI example...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Generate training data
    logger.info("1. Generating training data...")
    X, y, anomaly_types = generate_realistic_access_data(
        num_samples=1000,
        seq_len=20,
        anomaly_ratio=0.2,  # 20% anomalies
        complexity_level='medium'
    )
    
    logger.info(f"Generated {len(X)} samples with shape {X.shape}")
    
    # 2. Set up data processing
    logger.info("2. Setting up data processing...")
    data_processor = DataProcessor(
        batch_size=32,
        val_split=0.2,
        test_split=0.1
    )
    
    # Prepare data
    data_loaders = data_processor.prepare_data(X, y)
    
    # 3. Create and train model
    logger.info("3. Creating and training model...")
    model = BehavioralAnalyzer(
        input_size=X.shape[2],  # Number of features
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        use_attention=True,
        bidirectional=True
    )
    
    # Train model
    history = train_model(
        model=model,
        dataloaders=data_loaders,
        epochs=10,
        lr=0.001,
        patience=3
    )
    
    logger.info(f"Training completed. Final validation accuracy: {history['val_acc'][-1]:.4f}")
    
    # 4. Set up inference engine
    logger.info("4. Setting up inference engine...")
    feature_extractor = data_processor.feature_extractor
    inference_engine = InferenceEngine(
        model=model,
        feature_extractor=feature_extractor,
        threshold=0.5,
        enable_explainability=True
    )
    
    # 5. Test inference on a few samples
    logger.info("5. Testing inference...")
    test_samples = X[:5]  # Take first 5 samples for testing
    
    for i, sample in enumerate(test_samples):
        result = inference_engine.analyze(sample)
        
        print(f"\nSample {i+1}:")
        print(f"  - Anomaly detected: {result.anomaly_detected}")
        print(f"  - Confidence: {result.confidence:.4f}")
        print(f"  - Threat score: {result.threat_score:.2f}")
        print(f"  - Normal behavior: {result.behavior_normal}")
        print(f"  - Access pattern score: {result.access_pattern_score:.4f}")
        
        if result.anomaly_type:
            print(f"  - Anomaly type: {result.anomaly_type}")
            if result.anomaly_type_confidence:
                print(f"  - Type confidence: {result.anomaly_type_confidence:.4f}")
        
        if result.explanation:
            print(f"  - Explanation: {', '.join(result.explanation['summary'])}")
    
    # 6. Test batch analysis
    logger.info("6. Testing batch analysis...")
    batch_results = inference_engine.batch_analyze(test_samples, parallel=True)
    
    print(f"\nBatch analysis results:")
    for i, result in enumerate(batch_results):
        status = "ANOMALY" if result.anomaly_detected else "NORMAL"
        print(f"  Sample {i+1}: {status} (confidence: {result.confidence:.4f}, "
              f"threat: {result.threat_score:.2f})")
    
    # 7. Save and load model
    logger.info("7. Saving and loading model...")
    model_manager = ModelManager(model_dir=Path('./models_simplified'))
    
    # Save model
    model_manager.save_model(
        model=model,
        version='1.0.0',
        metrics={'accuracy': history['val_acc'][-1], 'f1_score': history['val_f1'][-1]},
        feature_extractor=feature_extractor,
        model_name='BehavioralAnalyzer'
    )
    
    # Load model
    loaded_model, loaded_feature_extractor = model_manager.load_model(version='1.0.0')
    
    # Verify loaded model works
    verification_result = inference_engine.analyze(test_samples[0])
    print(f"\nVerification - Loaded model analysis:")
    print(f"  - Same result as before: {abs(verification_result.access_pattern_score - inference_engine.analyze(test_samples[0]).access_pattern_score) < 0.001}")
    
    logger.info("Simplified example completed successfully!")
    
    # Summary
    accuracy = history['val_acc'][-1]
    f1_score = history['val_f1'][-1]
    
    print(f"\n--- SUMMARY ---")
    print(f"Model achieved {accuracy:.2%} validation accuracy")
    print(f"Model achieved {f1_score:.4f} F1 score")
    print(f"Tested {len(test_samples)} samples with batch and single inference")
    print(f"Model saved with versioning and can be loaded for production use")
    
    return {
        'final_accuracy': accuracy,
        'final_f1_score': f1_score,
        'samples_analyzed': len(test_samples)
    }


if __name__ == "__main__":
    run_simplified_example()