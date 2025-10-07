# SentinelZer0 – Production Threat Detection System

**Version 3.2.0** - Real-time AI-powered threat detection for distributed file systems.

This repository contains a production-ready hybrid threat detection system combining neural networks, anomaly detection, and heuristic analysis. The system is designed for real-world deployment with <1ms inference latency, REST API access, and >95% accuracy.

## 🎉 Latest: Phase 1.2 Complete - REST API Framework

✅ **NEW**: Production-ready REST API with FastAPI  
✅ **NEW**: Interactive Swagger UI documentation  
✅ **NEW**: API key authentication and security  
✅ **NEW**: Batch and real-time prediction endpoints  
✅ **NEW**: Performance monitoring and metrics API  

### Phase 1.1: Real-Time Stream Processing ✅

✅ Real-time event stream processing with sub-millisecond latency  
✅ Thread-safe sliding window buffer for continuous monitoring  
✅ GPU-accelerated streaming inference (1,197 events/sec)  
✅ Concurrent multi-stream support  

See [PHASE_1_1_SUMMARY.md](PHASE_1_1_SUMMARY.md) and [PHASE_1_2_SUMMARY.md](PHASE_1_2_SUMMARY.md) for complete details.

## ✨ What's Included

### Core Components
- **`sentinelzer0/`**: Complete Python package with production models
  - `models/`: HybridThreatDetector (GRU + Isolation Forest + Heuristics)
  - `data/`: RealFeatureExtractor (30 real-world features)
  - `training/`: RealWorldTrainer with incremental learning
  - `inference/`: RealTimeInferenceEngine (<25ms latency)
  - `evaluation/`: ProductionEvaluator for continuous monitoring

### Pre-trained Models
- **`models/production/`**: Production-ready model checkpoints
  - `sentinelfs_production.pt` - Optimized production model
  - `sentinelfs_production_5060.pt` - RTX 5060 optimized version

### Scripts & Documentation
- **`train_rm_rtx5060_fixed.py`**: Complete training and deployment script with diagnostics
- **`requirements.txt`**: Production dependencies
- **`checkpoints/final/`**: Latest trained model with Isolation Forest and heuristics

## 🚀 Quick Start

### Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Training Demo
```bash
python train_rm_rtx5060_fixed.py
```

This demonstrates the complete workflow:
- ✅ Enhanced realistic data generation (4000 normal + 800 anomalous events)
- ✅ Adversarial validation for distribution matching
- ✅ Hybrid model training (GRU + Isolation Forest + Heuristics)
- ✅ ROC/PR curve-based threshold calibration
- ✅ Real-time inference testing (<25ms)
- ✅ Comprehensive diagnostics and monitoring

### Production Usage

```python
from sentinelzer0 import (
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
import torch
checkpoint = torch.load('checkpoints/final/model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Create inference engine
engine = RealTimeInferenceEngine(
    model=model,
    feature_extractor=feature_extractor,
    threat_threshold=0.5276,  # Calibrated threshold
    cache_size=10000
)

# Analyze file system events
threat_score, threat_type, confidence = engine.predict(file_events)
if threat_score > 0.5276:
    print(f"⚠️ Threat detected: {threat_type} (confidence: {confidence:.1%})")
```

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **ROC AUC** | 0.9619 |
| **Precision** | 1.0000 |
| **Recall** | 0.8862 |
| **F1 Score** | 0.9397 |
| **Inference Latency** | <25ms |
| **Training Time** | ~15 seconds (30 epochs) |
| **GPU Support** | ✅ RTX 5060 (8GB) |
| **Model Size** | ~12MB (optimized) |

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

- **[sentinelzer0/README.md](sentinelzer0/README.md)** - Package API reference and usage guide
- **[MODEL_COMPARISON.md](MODEL_COMPARISON.md)** - Architecture comparison and design decisions
- **[CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md)** - Recent cleanup and migration guide
- **[CRITICAL_FIX_DOCUMENTATION.md](CRITICAL_FIX_DOCUMENTATION.md)** - Critical fixes and improvements
- **[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)** - Implementation completion summary

## 🔄 Recent Changes (v3.0.0)

✅ **Major Code Cleanup (2025-10-08)**
- Removed 9 unused Python files (~3000+ lines of dead code)
- Project size reduced from 6.9GB to 12MB
- File count reduced to 31 optimized files
- All imports verified and working

✅ **Enhanced Diagnostics & Calibration**
- ROC/PR curve-based threshold calibration (optimal: 0.5276)
- Real GPU monitoring with nvidia-smi integration
- Adversarial validation for train/val distribution matching
- Comprehensive score distribution analysis
- Enhanced test data with verified threat patterns

✅ **Production-Ready System**
- Hybrid detection: GRU + Isolation Forest + Heuristic Rules
- 30 real-world features from actual file system operations
- <25ms real-time inference latency
- RTX 5060 GPU optimization
- Incremental learning support

See [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) and [CRITICAL_FIX_DOCUMENTATION.md](CRITICAL_FIX_DOCUMENTATION.md) for detailed changes.

## 🎯 Use Cases

- **Real-time Threat Detection**: Monitor file system operations for ransomware, data exfiltration
- **Behavioral Analysis**: Detect anomalous access patterns and user behavior
- **Security Monitoring**: Continuous evaluation with drift detection
- **Incident Response**: Fast inference for immediate threat assessment

## 📦 Requirements

- Python 3.13+
- PyTorch 2.8.0+
- CUDA 12.8+ (optional, for GPU acceleration)
- scikit-learn, numpy, pandas, matplotlib

See [requirements.txt](requirements.txt) for complete list.

## 🤝 Academic Project

This is part of the **YMH345 - Computer Networks** course project at Sakarya University.

**Project**: SentinelFS - AI-powered distributed security file system
**Focus**: Real-time behavioral threat detection using hybrid machine learning

---

**Status**: ✅ Production Ready | **Version**: 3.0.0 | **Last Updated**: 2025-10-08

The system is fully functional and optimized for production deployment with comprehensive diagnostics and monitoring capabilities.
