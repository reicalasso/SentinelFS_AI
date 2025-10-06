#!/usr/bin/env python
"""
Quick test script for SentinelFS AI module
"""

import numpy as np
import torch
from sentinelfs_ai import (
    BehavioralAnalyzer,
    FeatureExtractor,
    InferenceEngine,
    generate_sample_data,
    train_model,
    AnomalyType
)

def main():
    print("=" * 60)
    print("🛡️  SentinelFS AI Module Test")
    print("=" * 60)
    
    # 1. Test model creation
    print("\n[1/5] Testing model creation...")
    input_size = 7  # Number of features
    model = BehavioralAnalyzer(
        input_size=input_size,
        hidden_size=64,
        num_layers=3,
        dropout=0.3
    )
    print(f"✓ Model created successfully")
    print(f"  - Input size: {input_size}")
    print(f"  - Hidden size: 64")
    print(f"  - Layers: 3")
    
    # 2. Test data generation
    print("\n[2/5] Testing data generation...")
    data, labels, _ = generate_sample_data(
        num_samples=100,
        seq_len=10,
        anomaly_ratio=0.2,
        include_anomaly_types=True
    )
    
    # Split into train/val
    split_idx = int(len(data) * 0.8)
    X_train, X_val = data[:split_idx], data[split_idx:]
    y_train, y_val = labels[:split_idx], labels[split_idx:]
    
    print(f"✓ Data generated successfully")
    print(f"  - Train samples: {len(X_train)}")
    print(f"  - Val samples: {len(X_val)}")
    print(f"  - Anomaly ratio: {y_train.mean():.2%}")
    
    # 3. Test feature extraction
    print("\n[3/5] Testing feature extractor...")
    feature_extractor = FeatureExtractor()
    # Fit the scaler on training data
    X_train_normalized = feature_extractor.fit_transform(X_train)
    X_val_normalized = feature_extractor.transform(X_val)
    print(f"✓ Feature extraction successful")
    print(f"  - Original shape: {X_train.shape}")
    print(f"  - Normalized train shape: {X_train_normalized.shape}")
    print(f"  - Normalized val shape: {X_val_normalized.shape}")
    
    # 4. Test quick training
    print("\n[4/5] Testing model training (5 epochs)...")
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_normalized),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_normalized),
        torch.FloatTensor(y_val)
    )
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=16, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=16)
    }
    
    history = train_model(
        model=model,
        dataloaders=dataloaders,
        epochs=5,
        lr=0.001,
        patience=10
    )
    
    print(f"✓ Training completed")
    print(f"  - Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  - Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  - Final val accuracy: {history['val_acc'][-1]:.4f}")
    print(f"  - Final val F1: {history['val_f1'][-1]:.4f}")
    
    # 5. Test inference
    print("\n[5/5] Testing inference engine...")
    engine = InferenceEngine(
        model=model,
        feature_extractor=feature_extractor,
        threshold=0.5,
        enable_explainability=True
    )
    
    # Analyze a sample sequence
    test_sequence = X_val[0]
    result = engine.analyze(test_sequence)
    
    print(f"✓ Inference successful")
    print(f"  - Anomaly detected: {result.anomaly_detected}")
    print(f"  - Confidence: {result.confidence:.2%}")
    print(f"  - Threat score: {result.threat_score:.2f}")
    print(f"  - Behavior normal: {result.behavior_normal}")
    
    if result.explanation:
        print(f"  - Top features: {list(result.explanation['important_features'].keys())[:3]}")
    
    # Test batch processing
    print("\n[Bonus] Testing batch inference...")
    batch_results = engine.batch_analyze(X_val[:5])
    anomalies_found = sum(1 for r in batch_results if r.anomaly_detected)
    print(f"✓ Batch processing successful")
    print(f"  - Processed: {len(batch_results)} sequences")
    print(f"  - Anomalies found: {anomalies_found}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed successfully!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("  1. Train with more data for better accuracy")
    print("  2. Save model: torch.save(model.state_dict(), 'model.pt')")
    print("  3. Integrate with Rust API via HTTP/gRPC")
    print("  4. Monitor performance in production")

if __name__ == '__main__':
    main()
