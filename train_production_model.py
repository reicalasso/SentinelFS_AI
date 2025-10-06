#!/usr/bin/env python
"""
Production-grade model training script for SentinelFS AI
Train a robust behavioral analyzer with comprehensive evaluation
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from sentinelfs_ai import (
    BehavioralAnalyzer,
    FeatureExtractor,
    InferenceEngine,
    generate_sample_data,
    train_model,
    evaluate_model,
    AnomalyType
)
from torch.utils.data import TensorDataset, DataLoader


def train_production_model(
    num_samples: int = 10000,
    seq_len: int = 20,
    anomaly_ratio: float = 0.3,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.0005,
    patience: int = 15,
    hidden_size: int = 128,
    num_layers: int = 4,
    dropout: float = 0.4
):
    """Train a production-ready model with best practices."""
    
    print("=" * 80)
    print("🛡️  SentinelFS Production Model Training")
    print("=" * 80)
    print(f"\n📅 Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n⚙️  Configuration:")
    print(f"  - Training samples: {num_samples}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Anomaly ratio: {anomaly_ratio:.1%}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - LSTM layers: {num_layers}")
    print(f"  - Dropout: {dropout}")
    
    # Create directories
    model_dir = Path('./models')
    checkpoint_dir = Path('./checkpoints')
    results_dir = Path('./results')
    model_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # Step 1: Generate comprehensive dataset
    print(f"\n{'='*80}")
    print("📊 Step 1/5: Generating Training Data")
    print("="*80)
    
    data, labels, anomaly_types = generate_sample_data(
        num_samples=num_samples,
        seq_len=seq_len,
        anomaly_ratio=anomaly_ratio,
        include_anomaly_types=True
    )
    
    # Split: 70% train, 15% val, 15% test
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.15)
    
    indices = np.random.permutation(len(data))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    X_train = data[train_idx]
    y_train = labels[train_idx]
    types_train = anomaly_types[train_idx]
    
    X_val = data[val_idx]
    y_val = labels[val_idx]
    types_val = anomaly_types[val_idx]
    
    X_test = data[test_idx]
    y_test = labels[test_idx]
    types_test = anomaly_types[test_idx]
    
    print(f"✓ Data generated successfully")
    print(f"  - Training set: {len(X_train)} samples ({y_train.mean():.1%} anomalies)")
    print(f"  - Validation set: {len(X_val)} samples ({y_val.mean():.1%} anomalies)")
    print(f"  - Test set: {len(X_test)} samples ({y_test.mean():.1%} anomalies)")
    
    # Distribution of anomaly types
    print(f"\n  Anomaly type distribution (train):")
    for i in range(5):
        count = (types_train == i).sum()
        print(f"    - {AnomalyType.get_name(i)}: {count} ({count/len(types_train)*100:.1f}%)")
    
    # Step 2: Feature extraction and normalization
    print(f"\n{'='*80}")
    print("🔧 Step 2/5: Feature Engineering")
    print("="*80)
    
    feature_extractor = FeatureExtractor()
    X_train_norm = feature_extractor.fit_transform(X_train)
    X_val_norm = feature_extractor.transform(X_val)
    X_test_norm = feature_extractor.transform(X_test)
    
    print(f"✓ Features normalized using StandardScaler")
    print(f"  - Feature dimensions: {X_train_norm.shape[2]}")
    print(f"  - Mean: {X_train_norm.mean():.4f}")
    print(f"  - Std: {X_train_norm.std():.4f}")
    
    # Step 3: Create model
    print(f"\n{'='*80}")
    print("🏗️  Step 3/5: Building Model Architecture")
    print("="*80)
    
    input_size = X_train_norm.shape[2]
    model = BehavioralAnalyzer(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_attention=True,
        bidirectional=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model architecture created")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"\n  Architecture:")
    print(f"    ├─ Bidirectional LSTM ({num_layers} layers)")
    print(f"    ├─ Self-Attention mechanism")
    print(f"    ├─ Layer Normalization")
    print(f"    ├─ Dropout ({dropout})")
    print(f"    ├─ Main classifier (binary)")
    print(f"    └─ Auxiliary classifier (4-way)")
    
    # Step 4: Training
    print(f"\n{'='*80}")
    print("🎯 Step 4/5: Training Model")
    print("="*80)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_norm),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_norm),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    # Train
    print(f"\n🚀 Starting training...")
    print(f"  (This may take several minutes on CPU, faster on GPU)")
    
    history = train_model(
        model=model,
        dataloaders=dataloaders,
        epochs=epochs,
        lr=learning_rate,
        patience=patience,
        checkpoint_dir=checkpoint_dir
    )
    
    print(f"\n✓ Training completed!")
    print(f"  - Best validation loss: {min(history['val_loss']):.4f}")
    print(f"  - Best validation accuracy: {max(history['val_acc']):.4f}")
    print(f"  - Best validation F1: {max(history['val_f1']):.4f}")
    
    # Step 5: Comprehensive evaluation
    print(f"\n{'='*80}")
    print("📈 Step 5/5: Model Evaluation")
    print("="*80)
    
    # Evaluate on test set
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_norm),
        torch.FloatTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loss, test_metrics = evaluate_model(model, test_loader, device)
    
    # Add ROC-AUC and confusion matrix
    from sklearn.metrics import roc_auc_score, confusion_matrix
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    test_metrics['roc_auc'] = roc_auc_score(all_labels, all_preds)
    pred_binary = (np.array(all_preds) >= 0.5).astype(int)
    test_metrics['confusion_matrix'] = confusion_matrix(all_labels, pred_binary)
    
    print(f"\n✓ Test Set Performance:")
    print(f"  - Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  - Precision: {test_metrics['precision']:.4f}")
    print(f"  - Recall: {test_metrics['recall']:.4f}")
    print(f"  - F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"  - ROC-AUC: {test_metrics['roc_auc']:.4f}")
    
    # Confusion matrix
    if 'confusion_matrix' in test_metrics:
        cm = test_metrics['confusion_matrix']
        print(f"\n  Confusion Matrix:")
        print(f"    ┌─────────────┬──────────┬──────────┐")
        print(f"    │             │ Pred: 0  │ Pred: 1  │")
        print(f"    ├─────────────┼──────────┼──────────┤")
        print(f"    │ Actual: 0   │ {cm[0,0]:6d}   │ {cm[0,1]:6d}   │  (Normal)")
        print(f"    │ Actual: 1   │ {cm[1,0]:6d}   │ {cm[1,1]:6d}   │  (Anomaly)")
        print(f"    └─────────────┴──────────┴──────────┘")
    
    # Test inference engine
    print(f"\n{'='*80}")
    print("🔬 Testing Inference Engine")
    print("="*80)
    
    engine = InferenceEngine(
        model=model,
        feature_extractor=feature_extractor,
        threshold=0.5,
        enable_explainability=True
    )
    
    # Test on some anomalies
    anomaly_indices = np.where(y_test == 1)[0][:5]
    print(f"\nTesting on {len(anomaly_indices)} known anomalies:")
    
    for i, idx in enumerate(anomaly_indices, 1):
        result = engine.analyze(X_test[idx])
        print(f"\n  [{i}] Test sample {idx}:")
        print(f"      ├─ Detected: {'✓ YES' if result.anomaly_detected else '✗ NO'}")
        print(f"      ├─ Confidence: {result.confidence:.1%}")
        print(f"      ├─ Threat Score: {result.threat_score:.1f}/100")
        if result.anomaly_type:
            print(f"      ├─ Type: {result.anomaly_type} ({result.anomaly_type_confidence:.1%})")
        if result.explanation and result.explanation['summary']:
            print(f"      └─ Reasons: {', '.join(result.explanation['summary'][:2])}")
    
    # Save model
    print(f"\n{'='*80}")
    print("💾 Saving Model")
    print("="*80)
    
    model_path = model_dir / f'behavioral_analyzer_production.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'use_attention': True,
            'bidirectional': True
        },
        'feature_extractor': feature_extractor,
        'training_metrics': test_metrics,
        'history': history,
        'timestamp': datetime.now().isoformat()
    }, model_path)
    
    print(f"✓ Model saved to: {model_path}")
    
    # Save training summary
    summary = {
        'training_config': {
            'num_samples': num_samples,
            'seq_len': seq_len,
            'anomaly_ratio': anomaly_ratio,
            'epochs': len(history['train_loss']),
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout
        },
        'model_stats': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        },
        'performance': {
            'test_accuracy': float(test_metrics['accuracy']),
            'test_precision': float(test_metrics['precision']),
            'test_recall': float(test_metrics['recall']),
            'test_f1': float(test_metrics['f1_score']),
            'test_roc_auc': float(test_metrics['roc_auc']),
            'best_val_loss': float(min(history['val_loss'])),
            'best_val_acc': float(max(history['val_acc'])),
            'best_val_f1': float(max(history['val_f1']))
        },
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = results_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Training summary saved to: {summary_path}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("🎉 Training Complete!")
    print("="*80)
    print(f"\n📊 Final Performance Summary:")
    print(f"  ├─ Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"  ├─ Test F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"  ├─ Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  └─ Model Size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    print(f"\n📁 Saved Files:")
    print(f"  ├─ Model: {model_path}")
    print(f"  ├─ Summary: {summary_path}")
    print(f"  └─ Checkpoints: {checkpoint_dir}/")
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Review training summary: cat {summary_path}")
    print(f"  2. Load model: torch.load('{model_path}')")
    print(f"  3. Deploy to production environment")
    print(f"  4. Monitor performance and retrain periodically")
    
    print(f"\n💡 Integration Example:")
    print(f"""
    from sentinelfs_ai import InferenceEngine, FeatureExtractor
    import torch
    
    # Load model
    checkpoint = torch.load('{model_path}')
    model = BehavioralAnalyzer(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create inference engine
    engine = InferenceEngine(
        model=model,
        feature_extractor=checkpoint['feature_extractor'],
        threshold=0.5
    )
    
    # Analyze file access
    result = engine.analyze(access_sequence)
    if result.anomaly_detected:
        print(f"⚠️  Threat detected! Score: {{result.threat_score}}")
    """)
    
    print(f"\n{'='*80}")
    print(f"✅ All done! Model is ready for production use.")
    print(f"{'='*80}\n")
    
    return model, feature_extractor, test_metrics, history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train production SentinelFS AI model')
    parser.add_argument('--samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--seq-len', type=int, default=20, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--layers', type=int, default=4, help='Number of LSTM layers')
    parser.add_argument('--quick', action='store_true', help='Quick training (1000 samples, 20 epochs)')
    
    args = parser.parse_args()
    
    if args.quick:
        print("🚀 Quick training mode enabled")
        args.samples = 1000
        args.epochs = 20
    
    train_production_model(
        num_samples=args.samples,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.layers
    )
