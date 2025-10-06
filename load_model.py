#!/usr/bin/env python
"""
Load and test the trained production model
"""

import torch
import numpy as np
from sentinelfs_ai import InferenceEngine, BehavioralAnalyzer

def load_model(model_path='models/behavioral_analyzer_production.pt'):
    """Load the trained production model."""
    print(f"📦 Loading model from: {model_path}")
    
    # PyTorch 2.8+ requires weights_only=False for custom classes
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Create model
    model = BehavioralAnalyzer(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create inference engine
    engine = InferenceEngine(
        model=model,
        feature_extractor=checkpoint['feature_extractor'],
        threshold=0.5,
        enable_explainability=True
    )
    
    print("✓ Model loaded successfully")
    print(f"\n📊 Model Info:")
    print(f"  - Config: {checkpoint['model_config']}")
    print(f"  - Trained: {checkpoint['timestamp']}")
    print(f"  - Test Accuracy: {checkpoint['training_metrics']['accuracy']:.2%}")
    print(f"  - Test F1: {checkpoint['training_metrics']['f1_score']:.4f}")
    print(f"  - ROC-AUC: {checkpoint['training_metrics']['roc_auc']:.4f}")
    
    return engine, checkpoint

def demo_inference(engine):
    """Demonstrate model inference."""
    print(f"\n{'='*60}")
    print("🔬 Running Demo Inference")
    print("="*60)
    
    # Generate sample data
    from sentinelfs_ai import generate_sample_data
    
    data, labels, types = generate_sample_data(
        num_samples=10,
        seq_len=20,
        anomaly_ratio=0.5,
        include_anomaly_types=True
    )
    
    print(f"\nAnalyzing {len(data)} sample sequences...")
    
    # Analyze each sequence
    for i in range(len(data)):
        result = engine.analyze(data[i])
        actual_label = "ANOMALY" if labels[i] == 1 else "NORMAL"
        predicted = "ANOMALY" if result.anomaly_detected else "NORMAL"
        match = "✓" if (labels[i] == 1) == result.anomaly_detected else "✗"
        
        print(f"\n[{i+1}] {match} Actual: {actual_label} | Predicted: {predicted}")
        print(f"    ├─ Confidence: {result.confidence:.1%}")
        print(f"    ├─ Threat Score: {result.threat_score:.1f}/100")
        
        if result.anomaly_detected:
            print(f"    ├─ Type: {result.anomaly_type}")
            if result.explanation and result.explanation['summary']:
                print(f"    └─ Reasons: {', '.join(result.explanation['summary'][:2])}")

if __name__ == '__main__':
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/behavioral_analyzer_production.pt'
    
    try:
        engine, checkpoint = load_model(model_path)
        demo_inference(engine)
        
        print(f"\n{'='*60}")
        print("✅ Model is ready for production use!")
        print("="*60)
        
    except FileNotFoundError:
        print(f"❌ Error: Model file not found: {model_path}")
        print(f"\n💡 Train a model first:")
        print(f"   python train_production_model.py --quick")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
