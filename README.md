# SentinelFS AI – Production Threat Detection System

**Version 2.0.0** - Real-time AI-powered threat detection for distributed file systems.

This repository contains a production-ready hybrid threat detection system combining neural networks, anomaly detection, and heuristic analysis. The system is designed for real-world deployment with <25ms inference latency and >95% accuracy.

## ✨ What's Included

### Core Components
- **`sentinelfs_ai/`**: Complete Python package with production models
  - `models/`: HybridThreatDetector (GRU + Isolation Forest + Heuristics)
  - `data/`: RealFeatureExtractor (30 real-world features)
  - `training/`: RealWorldTrainer with incremental learning
  - `inference/`: RealTimeInferenceEngine (<25ms latency)
  - `evaluation/`: ProductionEvaluator for continuous monitoring
  - `management/`: Model lifecycle management tools

### Reference Models
- **`models/`**: Pre-trained production checkpoints
  - `behavioral_analyzer_production.pt` - Production hybrid model
  - `best_individual_model.pt` - Best single model
  - `ensemble_model/` - Ensemble configuration and components

### Scripts & Documentation
- **`train_real_model.py`**: Complete training and deployment demo
- **`REAL_MODEL_README.md`**: Comprehensive system documentation
- **`MODEL_COMPARISON.md`**: Performance comparison and architecture details
- **`CLEANUP_SUMMARY.md`**: Recent cleanup and migration notes
- **`requirements.txt`**: Production dependenciesS AI – Production Behavioral Analyzer

This repository now contains only the production-ready SentinelFS AI inference stack and the reference model artifacts. Training utilities, experimental datasets, and development notebooks have been removed to deliver a lean deployment package.

## Whats Included

- `sentinelfs_ai/`: Python package with the behavioral analyzer, inference engine, model management helpers, and supporting data utilities
- `models/`: Reference checkpoints for the production ensemble (including `behavioral_analyzer_production.pt`)
- `load_model.py`: Simple script for loading the packaged model and running a quick inference demo
- `MODEL_REPORT.md`: Detailed performance report for the shipped model release (v1.0.0)
- `requirements.txt`: Minimal runtime dependencies required for inference and export workflows

## 🚀 Quick Start

### Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Training Demo
```bash
python train_real_model.py
```

This demonstrates the complete workflow:
- ✅ Feature extraction (30 real-world features)
- ✅ Hybrid model training (GRU + Isolation Forest + Heuristics)
- ✅ Real-time inference testing (<25ms)
- ✅ Production monitoring and evaluation

### Production Usage

```python
from sentinelfs_ai import (
    HybridThreatDetector,
    RealFeatureExtractor,
    RealTimeInferenceEngine,
    ProductionEvaluator
)

# Initialize components
feature_extractor = RealFeatureExtractor()
model = HybridThreatDetector(
    input_size=30,
    hidden_size=128,
    num_layers=2,
    dropout=0.3
)

# Load trained model
checkpoint = torch.load('checkpoints/final/model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Create inference engine
engine = RealTimeInferenceEngine(
    model=model,
    feature_extractor=feature_extractor,
    threat_threshold=0.7,
    cache_size=10000
)

# Analyze file system events
threat_score, threat_type, confidence = engine.predict(file_events)
if threat_score > 0.7:
    print(f"⚠️ Threat detected: {threat_type} (confidence: {confidence:.1%})")
```

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 97.4% |
| **F1 Score** | 91.5% |
| **Inference Latency** | <25ms |
| **Training Time** | ~8 seconds (20 epochs) |
| **GPU Support** | ✅ CUDA enabled |

## 🏗️ Architecture

**HybridThreatDetector** - Three-component ensemble:
- **40%** GRU Neural Network - Sequential pattern learning
- **30%** Isolation Forest - Unsupervised anomaly detection
- **30%** Heuristic Rules - Domain-specific threat indicators

**Features** - 30 real-world indicators:
- Temporal patterns (access frequency, time-of-day)
- File characteristics (size, type, entropy)
- Behavioral signals (velocity, modification rate)
- Security indicators (ransomware patterns, suspicious extensions)

## 📚 Documentation

- **[REAL_MODEL_README.md](REAL_MODEL_README.md)** - Complete technical documentation
- **[MODEL_COMPARISON.md](MODEL_COMPARISON.md)** - Architecture comparison and design decisions
- **[CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md)** - Recent cleanup and migration guide
- **[sentinelfs_ai/README.md](sentinelfs_ai/README.md)** - Package API reference

## 🔄 Recent Changes (v2.0.0)

✅ **Production-Ready System**
- Removed 8 deprecated modules (~3000+ lines of synthetic code)
- Streamlined to production components only
- All imports verified and working
- Complete end-to-end testing passed

✅ **Real-World Focus**
- 30 real features from actual file system operations
- Hybrid detection (neural + anomaly + heuristics)
- <25ms real-time inference
- Incremental learning support

See [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) for detailed changes.

## 🎯 Use Cases

- **Real-time Threat Detection**: Monitor file system operations for ransomware, data exfiltration
- **Behavioral Analysis**: Detect anomalous access patterns and user behavior
- **Security Monitoring**: Continuous evaluation with drift detection
- **Incident Response**: Fast inference for immediate threat assessment

## 📦 Requirements

- Python 3.8+
- PyTorch 2.0.0+
- CUDA (optional, for GPU acceleration)
- scikit-learn, numpy, pandas

See [requirements.txt](requirements.txt) for complete list.

## 🤝 Academic Project

This is part of the **YMH345 - Computer Networks** course project at Sakarya University.

**Project**: SentinelFS - AI-powered distributed security file system
**Focus**: Real-time behavioral threat detection using hybrid machine learning

---

**Status**: ✅ Production Ready | **Version**: 2.0.0 | **Last Updated**: 2025-10-08

If you need to retrain or regenerate datasets, clone the original full repository history (prior to this cleanup) or author your own training pipeline using the modules in `sentinelfs_ai.training`.
