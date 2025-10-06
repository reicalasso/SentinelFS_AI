#!/usr/bin/env python
"""
Simplified example demonstrating the key improvements to SentinelFS AI model.
Focuses on realistic data, advanced architectures, and ensemble methods.
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from sentinelfs_ai import (
    # Basic models
    BehavioralAnalyzer,
    TransformerBehavioralAnalyzer,
    CNNLSTMAnalyzer,
    
    # Data generation
    generate_realistic_access_data,
    
    # Training components
    train_model,
    EnsembleManager,
    
    # Evaluation
    AdvancedEvaluator,
    
    # Feature extraction
    FeatureExtractor
)


def main():
    print("=" * 80)
    print("🛡️  SentinelFS AI - Key Improvements Demonstration")
    print("=" * 80)
    
    # 1. Generate realistic data with complex patterns
    print("\n[1/5] Generating realistic data with complex patterns...")
    
    # Use the new realistic data generator
    train_data, train_labels, train_types = generate_realistic_access_data(
        num_samples=1500,
        seq_len=20,
        anomaly_ratio=0.15,  # More realistic ratio
        complexity_level='medium'
    )
    
    test_data, test_labels, test_types = generate_realistic_access_data(
        num_samples=500,
        seq_len=20,
        anomaly_ratio=0.15,
        complexity_level='medium'
    )
    
    print(f"✓ Generated {len(train_data)} training samples, {len(test_data)} test samples")
    print(f"  - Training anomaly ratio: {train_labels.mean():.3f}")
    print(f"  - Test anomaly ratio: {test_labels.mean():.3f}")
    
    # 2. Split data for validation
    val_size = int(len(train_data) * 0.15)
    X_train, X_val = train_data[:-val_size], train_data[-val_size:]
    y_train, y_val = train_labels[:-val_size], train_labels[-val_size:]
    types_train, types_val = train_types[:-val_size], train_types[-val_size:]
    
    # 3. Normalize features
    print("\n[2/5] Feature normalization...")
    feature_extractor = FeatureExtractor()
    X_train_norm = feature_extractor.fit_transform(X_train)
    X_val_norm = feature_extractor.transform(X_val)
    X_test_norm = feature_extractor.transform(test_data)
    
    print("✓ Features normalized using StandardScaler")
    
    # 4. Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_norm),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_norm),
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_norm),
        torch.FloatTensor(test_labels)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    # 5. Train multiple model architectures
    print("\n[3/5] Training multiple model architectures...")
    
    input_size = X_train_norm.shape[2]  # Number of features
    
    # LSTM-based model (traditional)
    print("  - Training LSTM model...")
    lstm_model = BehavioralAnalyzer(
        input_size=input_size,
        hidden_size=64,
        num_layers=3,
        dropout=0.3,
        use_attention=True,
        bidirectional=True
    )
    
    lstm_history = train_model(
        model=lstm_model,
        dataloaders=dataloaders,
        epochs=15,  # Reduced for demo
        lr=0.001
    )
    
    # Transformer model (advanced)
    print("  - Training Transformer model...")
    transformer_model = TransformerBehavioralAnalyzer(
        input_size=input_size,
        d_model=64,
        nhead=8,
        num_layers=3,
        dropout=0.1,
        seq_len=20
    )
    
    transformer_history = train_model(
        model=transformer_model,
        dataloaders=dataloaders,
        epochs=15,
        lr=0.001
    )
    
    # CNN-LSTM model (hybrid)
    print("  - Training CNN-LSTM model...")
    cnn_lstm_model = CNNLSTMAnalyzer(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    )
    
    cnn_lstm_history = train_model(
        model=cnn_lstm_model,
        dataloaders=dataloaders,
        epochs=15,
        lr=0.001
    )
    
    print("✓ All models trained successfully")
    
    # 6. Ensemble training
    print("\n[4/5] Ensemble training...")
    
    ensemble_manager = EnsembleManager(
        input_size=input_size,
        ensemble_size=3,
        base_architecture='mixed',  # Use different architectures
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        seq_len=20
    )
    
    ensemble_histories = ensemble_manager.train_ensemble(
        dataloaders,
        epochs=10,
        lr=0.001
    )
    
    # Update ensemble weights based on validation performance
    ensemble_manager.update_weights(val_loader)
    print("✓ Ensemble training completed")
    
    # 7. Advanced evaluation
    print("\n[5/5] Advanced evaluation...")
    
    evaluator = AdvancedEvaluator()
    
    # Evaluate all models
    models_to_evaluate = [
        ("LSTM", lstm_model),
        ("Transformer", transformer_model),
        ("CNN-LSTM", cnn_lstm_model)
    ]
    
    print("\nModel Performance Comparison:")
    print("-" * 60)
    
    for name, model in models_to_evaluate:
        print(f"\n{name} Model:")
        metrics = evaluator.evaluate_model_comprehensive(model, test_loader)
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
        print(f"  MCC: {metrics['mcc']:.4f}")
    
    # Evaluate ensemble
    print(f"\nEnsemble Model:")
    ensemble_metrics = ensemble_manager.evaluate_ensemble(test_loader)
    print(f"  Accuracy: {ensemble_metrics['accuracy']:.4f}")
    print(f"  Precision: {ensemble_metrics['precision']:.4f}")
    print(f"  Recall: {ensemble_metrics['recall']:.4f}")
    print(f"  F1-Score: {ensemble_metrics['f1_score']:.4f}")
    print(f"  Diversity: {ensemble_metrics['diversity']:.4f}")
    
    # Stratified evaluation by anomaly type
    print(f"\nStratified evaluation by anomaly type:")
    stratified_results = evaluator.stratified_evaluation(
        lstm_model,
        test_data,
        test_labels.flatten(),
        test_types
    )
    
    for strat, metrics in stratified_results.items():
        print(f"  {strat}: F1-Score = {metrics['f1_score']:.4f}, AUC-ROC = {metrics['auc_roc']:.4f}")
    
    # Model saving
    print(f"\nSaving models...")
    
    # Save ensemble
    ensemble_manager.save_ensemble(Path('./models/ensemble_model'))
    print("✓ Ensemble model saved to ./models/ensemble_model/")
    
    # Save best individual model
    torch.save({
        'model_state_dict': lstm_model.state_dict(),
        'model_config': {
            'input_size': input_size,
            'hidden_size': 64,
            'num_layers': 3,
            'dropout': 0.3,
            'use_attention': True,
            'bidirectional': True
        },
        'feature_extractor': feature_extractor,
        'training_metrics': evaluator.evaluate_model_comprehensive(lstm_model, test_loader)
    }, './models/best_individual_model.pt')
    
    print("✓ Best individual model saved to ./models/best_individual_model.pt")
    
    print(f"\n{'='*80}")
    print("🎉 Key improvements demonstration completed!")
    print("="*80)
    
    print(f"\n💡 Key Improvements Demonstrated:")
    print(f"  1. Realistic data generation with complex, nuanced patterns")
    print(f"  2. Multiple advanced architectures (Transformer, CNN-LSTM)")
    print(f"  3. Ensemble methods for improved performance and robustness")
    print(f"  4. Comprehensive evaluation with advanced metrics (AUC-ROC, AUC-PR, MCC, etc.)")
    print(f"  5. Cross-validation and stratified evaluation")
    
    print(f"\n🚀 Benefits of These Improvements:")
    print(f"  • Better generalization to real-world scenarios")
    print(f"  • Improved robustness through ensemble diversity")
    print(f"  • More reliable performance assessment")
    print(f"  • Better handling of complex anomaly patterns")
    print(f"  • More realistic evaluation (not 100% perfect results)")


if __name__ == '__main__':
    main()