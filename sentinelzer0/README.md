# 🛡️ SentinelZer0 - Production Threat Detection Package

**Real-time AI-powered behavioral analysis for distributed file system security**

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE)

## 🎯 Overview

SentinelZer0 is a production-ready hybrid threat detection system that combines deep learning, anomaly detection, and heuristic rules for real-time file system security monitoring. The system achieves exceptional accuracy with sub-25ms inference latency.

### ✨ Key Features

- **🎯 High-Accuracy Detection**: ROC AUC 0.9619, F1 Score 0.9397
- **⚡ Real-time Inference**: <25ms latency with GPU optimization
- **🧠 Hybrid Architecture**: GRU Neural Network + Isolation Forest + Heuristic Rules
- **🔍 Explainable AI**: Feature importance and confidence scores
- **📊 Real-World Features**: 30 indicators from actual file system operations
- **🚀 Production-Ready**: Comprehensive diagnostics, monitoring, and calibration
- **🛡️ Adversarial Robustness**: Distribution validation and threshold calibration
- **📈 Continuous Monitoring**: Production evaluator with drift detection

## 🏗️ Architecture

### HybridThreatDetector Architecture
```
┌─────────────────────────────────────────────┐
│           Input: File System Events          │
│        (batch, seq_len=64, features=30)      │
└──────────────────┬──────────────────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  RealFeatureExtractor       │
    │  (30 real-world features)   │
    └──────────────┬──────────────┘
                   │
    ┌─────────┬─────────┬─────────┐
    │   GRU   │Isolation│Heuristic│
    │ Network │ Forest  │ Rules   │
    │ (40%)   │ (30%)   │ (30%)   │
    └─────────┴─────────┴─────────┘
                   │
         ┌─────────▼─────────┐
         │ Ensemble Fusion   │
         │ (Weighted Average)│
         └─────────┬─────────┘
                   │
    ┌──────────────▼──────────────┐
    │  Threat Classification       │
    │  (Score + Type + Confidence) │
    └─────────────────────────────┘
```

### Model Statistics

- **Total Parameters**: ~45K (GRU) + Isolation Forest
- **Model Size**: ~12MB (optimized checkpoint)
- **GPU Memory**: ~500MB during training, ~50MB inference
- **Training Time**: ~15 seconds (30 epochs)
- **Inference Latency**: <25ms on RTX 5060

## 🚀 Quick Start

### Basic Usage

```python
from sentinelzer0 import HybridThreatDetector, RealFeatureExtractor

# Initialize components
feature_extractor = RealFeatureExtractor()
model = HybridThreatDetector(
    input_size=30,
    hidden_size=128,
    num_layers=2,
    dropout=0.3
)

# Load pre-trained model
import torch
checkpoint = torch.load('../checkpoints/final/model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Process file system events
events = [...]  # Your file system events
features = feature_extractor.extract_features(events)
threat_score = model(features)
```

### Production Inference Engine

```python
from sentinelzer0 import RealTimeInferenceEngine

# Create production inference engine
engine = RealTimeInferenceEngine(
    model=model,
    feature_extractor=feature_extractor,
    threat_threshold=0.5276,  # Calibrated threshold
    cache_size=10000
)

# Real-time analysis
threat_score, threat_type, confidence = engine.predict(events)
if threat_score > 0.5276:
    print(f"🚨 THREAT DETECTED: {threat_type} (confidence: {confidence:.1%})")
```

### Training Pipeline

```python
from sentinelzer0 import RealWorldTrainer

# Initialize trainer
trainer = RealWorldTrainer(
    model=model,
    feature_extractor=feature_extractor,
    sequence_length=64,
    batch_size=64
)

# Train with diagnostics
training_results = trainer.train_with_diagnostics(
    train_events=train_events,
    val_events=val_events,
    epochs=30,
    enable_gpu=True
)

# Save trained model
trainer.save_model('../models/production/sentinelfs_production.pt')
```

## 📚 API Reference

### Core Classes

#### `HybridThreatDetector`
Main threat detection model combining GRU, Isolation Forest, and heuristics.

```python
model = HybridThreatDetector(
    input_size=30,      # Number of features
    hidden_size=128,    # GRU hidden size
    num_layers=2,       # GRU layers
    dropout=0.3         # Dropout rate
)
```

#### `RealFeatureExtractor`
Extracts 30 real-world features from file system events.

```python
extractor = RealFeatureExtractor()
features = extractor.extract_features(events)
```

#### `RealTimeInferenceEngine`
Production-ready inference engine with caching and optimization.

```python
engine = RealTimeInferenceEngine(
    model=model,
    feature_extractor=extractor,
    threat_threshold=0.5276,
    cache_size=10000
)
```

#### `RealWorldTrainer`
Complete training pipeline with diagnostics and monitoring.

```python
trainer = RealWorldTrainer(
    model=model,
    feature_extractor=extractor,
    sequence_length=64,
    batch_size=64
)
```

#### `ProductionEvaluator`
Continuous monitoring and evaluation for production deployment.

```python
evaluator = ProductionEvaluator()
metrics = evaluator.evaluate_model(model, test_events)
```

### Data Types

#### `AnalysisResult`
```python
@dataclass
class AnalysisResult:
    threat_score: float      # 0.0 to 1.0
    threat_type: AnomalyType # NORMAL, RANSOMWARE, etc.
    confidence: float        # 0.0 to 1.0
    features_used: int       # Number of features processed
```

#### `AnomalyType`
```python
class AnomalyType(Enum):
    NORMAL = "normal"
    RANSOMWARE = "ransomware"
    EXFILTRATION = "exfiltration"
    MALICIOUS_ACTIVITY = "malicious_activity"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
```

## 📊 Performance Metrics

### Latest Results (v3.0.0)
- **ROC AUC**: 0.9619
- **Precision**: 1.0000 (no false positives)
- **Recall**: 0.8862
- **F1 Score**: 0.9397
- **Inference Latency**: <25ms
- **GPU Utilization**: RTX 5060 (46% during training)

### Feature Importance (Top 10)
1. `rapid_modifications` - High-frequency file changes
2. `file_size_entropy` - File size variation patterns
3. `time_based_patterns` - Unusual timing patterns
4. `extension_changes` - Suspicious file extensions
5. `access_velocity` - Rapid access patterns
6. `directory_depth` - File system traversal patterns
7. `operation_types` - Operation type distributions
8. `user_behavior` - User activity patterns
9. `network_indicators` - Network-related patterns
10. `system_calls` - System call patterns

## 🔧 Configuration

### Model Configuration
```python
model_config = {
    'input_size': 30,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 64,
    'sequence_length': 64
}
```

### Training Configuration
```python
training_config = {
    'epochs': 30,
    'patience': 10,
    'min_delta': 0.001,
    'validation_split': 0.2,
    'enable_gpu': True,
    'save_best_only': True
}
```

### Inference Configuration
```python
inference_config = {
    'threat_threshold': 0.5276,  # Calibrated threshold
    'cache_size': 10000,
    'batch_size': 32,
    'enable_gpu': True,
    'warmup_iterations': 100
}
```

## 📁 Package Structure

```
```

sentinelzer0/
├── __init__.py              # Package initialization
├── data_types.py            # Core data structures
├── data/                    # Data processing modules
│   ├── __init__.py
│   ├── real_feature_extractor.py
│   └── data_processor.py    # Legacy (unused)
├── models/                  # Neural network models
│   ├── __init__.py
│   └── hybrid_detector.py
├── training/                # Training utilities
│   ├── __init__.py
│   └── real_trainer.py
├── inference/               # Inference engines
│   ├── __init__.py
│   └── real_engine.py
├── evaluation/              # Evaluation tools
│   ├── __init__.py
│   └── production_evaluator.py
└── utils/                   # Utilities
    ├── __init__.py
    └── logger.py
```

## 🔄 Recent Updates

### v3.0.0 (2025-10-08)
- ✅ Major code cleanup: Removed 9 unused files, reduced size to 12MB
- ✅ Enhanced diagnostics: ROC/PR calibration, adversarial validation
- ✅ GPU optimization: RTX 5060 support with real monitoring
- ✅ Production hardening: Comprehensive error handling and logging
- ✅ Performance improvements: Optimized inference pipeline

### v2.0.0 (2025-10-07)
- ✅ Real-world features: 30 indicators from actual file operations
- ✅ Hybrid architecture: GRU + Isolation Forest + Heuristics
- ✅ Production evaluator: Continuous monitoring capabilities
- ✅ Incremental learning: Support for model updates

## 🤝 Contributing

This is part of the **YMH345 - Computer Networks** course project at Sakarya University.

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd sentinelzer0

# Install dependencies
pip install -r ../requirements.txt

# Run tests
python ../train_rm_rtx5060_fixed.py
```

### Code Quality
- Type hints required for all functions
- Comprehensive docstrings
- Unit tests for critical components
- GPU compatibility testing

## 📄 License

MIT License - see [LICENSE](../LICENSE) file for details.

## � Support

For questions or issues:
- Check [CRITICAL_FIX_DOCUMENTATION.md](../CRITICAL_FIX_DOCUMENTATION.md)
- Review [IMPLEMENTATION_COMPLETE.md](../IMPLEMENTATION_COMPLETE.md)
- Run diagnostics: `python ../train_rm_rtx5060_fixed.py`

---

**Status**: ✅ Production Ready | **Version**: 3.0.0 | **Last Updated**: 2025-10-08

### Installation

```bash
# 1. Clone repository
git clone https://github.com/reicalasso/SentinelFS.git
cd SentinelFS

# 2. Set up virtual environment
python -m venv venv_sentinel
source venv_sentinel/bin/activate  # On Windows: venv_sentinel\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
# Or install core dependencies:
pip install torch scikit-learn numpy matplotlib seaborn

# 4. Verify installation
python -c "from sentinelzer0 import BehavioralAnalyzer; print('✓ Ready!')"
```

### Training

```bash
# Quick training (1K samples, 20 epochs, ~10 seconds)
python train_production_model.py --quick

# Full production training (10K samples, 100 epochs, ~5 minutes)
python train_production_model.py

# Custom training with advanced options
python train_production_model.py \
  --samples 5000 \
  --epochs 50 \
  --batch-size 128 \
  --lr 0.001 \
  --hidden-size 256 \
  --layers 6 \
  --architecture mixed \
  --adversarial-training \
  --ensemble-size 5
```

### Inference Examples

#### Single Sequence Analysis
```python
import torch
import numpy as np
from sentinelzer0 import InferenceEngine, BehavioralAnalyzer, ModelManager

# Load trained model using ModelManager
model_manager = ModelManager()
model, feature_extractor = model_manager.load_model(version="latest")

# Create inference engine
engine = InferenceEngine(
    model=model,
    feature_extractor=feature_extractor,
    threshold=0.5,
    enable_explainability=True
)

# Analyze file access sequence (20 timesteps, 7 features each)
access_sequence = np.random.randn(20, 7)  # Replace with real data
result = engine.analyze(access_sequence)

# Check result
if result.anomaly_detected:
    print(f"⚠️  THREAT DETECTED!")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   Threat Score: {result.threat_score:.1f}/100")
    print(f"   Type: {result.anomaly_type}")
    print(f"   Type Confidence: {result.anomaly_type_confidence:.1%}")
    if result.explanation:
        print(f"   Reasons: {', '.join(result.explanation['summary'])}")
else:
    print(f"✓ Normal behavior (confidence: {result.confidence:.1%})")
```

#### Batch Processing
```python
# Analyze multiple sequences efficiently
sequences = [access_seq1, access_seq2, access_seq3]  # List of numpy arrays (seq_len, num_features)
results = engine.batch_analyze(sequences, parallel=True)

for i, result in enumerate(results):
    status = "ANOMALY" if result.anomaly_detected else "NORMAL"
    print(f"Sequence {i}: {status} (score: {result.threat_score:.1f}, conf: {result.confidence:.1%})")
```

#### Ensemble Model Usage
```python
from sentinelzer0 import EnsembleManager

# Create ensemble with different architectures
ensemble_manager = EnsembleManager(
    input_size=7,
    ensemble_size=5,
    base_architecture='mixed',  # lstm, transformer, cnn-lstm, or mixed
    hidden_size=128,
    num_layers=3
)

# Train ensemble
histories = ensemble_manager.train_ensemble(dataloaders, epochs=50)

# Make predictions with ensemble
ensemble_pred, individual_preds = ensemble_manager.predict(torch.tensor(access_sequence).unsqueeze(0))
```

## 🔧 Configuration

### Feature Vector (7 dimensions)
```python
features = [
    'file_size_mb',        # Size of accessed file (MB)
    'access_hour',         # Hour of day (0-23)
    'access_type',         # 0=read, 1=write, 2=delete, 3=rename
    'day_of_week',         # 0=Monday, 6=Sunday
    'access_frequency',    # Accesses per hour
    'file_category',       # 0=document, 1=code, 2=media, 3=system, 4=other
    'access_velocity'      # Rate of change in access (files per minute)
]
```

### Training Configuration
```python
from sentinelzer0 import TrainingConfig

config = TrainingConfig(
    num_samples=5000,        # Total training samples
    seq_len=20,              # Sequence length
    anomaly_ratio=0.2,       # Proportion of anomalous samples
    batch_size=64,           # Training batch size
    epochs=100,              # Maximum training epochs
    learning_rate=0.001,     # Initial learning rate
    patience=15,             # Early stopping patience
    hidden_size=128,         # LSTM hidden units
    num_layers=4,            # LSTM layers
    dropout=0.3,             # Dropout rate
    model_dir='./models',     # Model save directory
    checkpoint_dir='./checkpoints'  # Checkpoint directory
)
```

### Advanced Hyperparameters
```python
advanced_config = {
    'seq_len': 20,            # Sequence length (timesteps)
    'input_size': 7,          # Number of features
    'hidden_size': 128,       # LSTM/Transformer hidden units
    'num_layers': 4,          # Number of layers
    'dropout': 0.3,           # Dropout for regularization
    'batch_size': 64,         # Training batch size
    'learning_rate': 0.001,   # Initial learning rate
    'weight_decay': 1e-5,     # L2 regularization
    'patience': 15,           # Early stopping patience
    'threshold': 0.5,         # Classification threshold
    'adversarial_ratio': 0.3, # Ratio of adversarial examples during training
    'epsilon': 0.01,          # Adversarial perturbation magnitude
    'ensemble_size': 5,       # Number of models in ensemble
    'base_architecture': 'mixed'  # 'lstm', 'transformer', 'cnn-lstm', 'mixed'
}
```

## 📊 Anomaly Types

| Type ID | Name | Description | Detection Characteristics |
|---------|------|-------------|---------------------------|
| 0 | Normal | Standard access patterns | Standard business hours, regular patterns |
| 1 | Data Exfiltration | Large off-hours transfers | Large files, off-hours, high velocity |
| 2 | Ransomware | Rapid file modifications/encryptions | High frequency, sequential access |
| 3 | Privilege Escalation | Unusual admin access patterns | System files, admin hours, access type |
| 4 | Other Anomaly | Unclassified suspicious activity | Deviates from learned normal patterns |

## 🔬 Advanced Features

### Adversarial Training
```python
from sentinelzer0 import AdversarialTrainer, RobustnessEvaluator

# Create adversarial trainer
adv_trainer = AdversarialTrainer(
    model=model,
    dataloaders=dataloaders,
    adversarial_ratio=0.3,  # 30% adversarial examples
    epsilon=0.01           # Perturbation magnitude
)

# Train with adversarial examples
history = adv_trainer.train()

# Evaluate model robustness
evaluator = RobustnessEvaluator(model)
robustness_results = evaluator.evaluate_robustness(test_data, test_labels)
```

### Multi-Model Ensemble
```python
from sentinelzer0 import EnsembleManager

# Create diverse ensemble
ensemble_mgr = EnsembleManager(
    input_size=7,
    ensemble_size=7,              # Number of models
    base_architecture='mixed',    # Mix of architectures for diversity
    hidden_size=128,
    num_layers=3,
    dropout=0.3
)

# Train and evaluate ensemble
histories = ensemble_mgr.train_ensemble(dataloaders, epochs=50)
ensemble_metrics = ensemble_mgr.evaluate_ensemble(test_loader)

print(f"Ensemble Accuracy: {ensemble_metrics['accuracy']:.4f}")
print(f"Ensemble F1 Score: {ensemble_metrics['f1_score']:.4f}")
print(f"Model Diversity: {ensemble_metrics['diversity']:.4f}")
```

### Advanced Evaluation
```python
from sentinelzer0 import AdvancedEvaluator

evaluator = AdvancedEvaluator()

# Comprehensive model evaluation
comprehensive_metrics = evaluator.evaluate_model_comprehensive(model, test_loader)

# Cross-validation
cv_results = evaluator.cross_validate(
    model_class=BehavioralAnalyzer,
    model_params={'input_size': 7, 'hidden_size': 128, 'num_layers': 3},
    data=test_data,
    labels=test_labels,
    n_folds=5
)

# Stratified evaluation by anomaly type
stratified_results = evaluator.stratified_evaluation(
    model=model,
    data=test_data,
    labels=test_labels,
    anomaly_types=anomaly_type_labels
)
```

## 📈 Advanced Training & Evaluation

---

**Status**: ✅ Production Ready | **Version**: 3.0.0 | **Last Updated**: 2025-10-08

### Performance Benchmarking
```python
# Benchmark model performance
perf_metrics = model_manager.benchmark_model(
    model=model,
    input_shape=(1, 20, 7),  # (batch_size, seq_len, features)
    num_iterations=1000
)

print(f"Average Latency: {perf_metrics['avg_latency_ms']:.2f}ms")
print(f"Throughput: {perf_metrics['throughput_per_sec']:.2f} inferences/sec")
```

## 🧪 Testing

```bash
# Run comprehensive tests
python test_ai_module.py

# Run system scan tests
python test_system_scan.py

# Expected comprehensive output:
# ✓ Model creation successful
# ✓ Data generation successful  
# ✓ Feature extraction successful
# ✓ Training completed (Val F1: ~0.98+)
# ✓ Inference successful
# ✓ Batch processing successful
# ✓ Adversarial training successful
# ✓ Ensemble training successful
# ✓ Model export successful
# ✓ Performance benchmarking successful
```

### Run with specific test modules:
```bash
# Test data generation
python -m pytest test_ai_module.py::test_data_generation -v

# Test model training
python -m pytest test_ai_module.py::test_training -v

# Test adversarial robustness
python -m pytest test_ai_module.py::test_adversarial -v
```

## 📁 Package Structure

```
sentinelzer0/
├── __init__.py              # Package initialization
├── data_types.py            # Core data structures
├── README.md                # This documentation
├── data/                    # Data processing modules
│   ├── __init__.py
│   └── real_feature_extractor.py
├── models/                  # Neural network models
│   ├── __init__.py
│   └── hybrid_detector.py
├── training/                # Training utilities
│   ├── __init__.py
│   └── real_trainer.py
├── inference/               # Inference engines
│   ├── __init__.py
│   └── real_engine.py
├── evaluation/              # Evaluation tools
│   ├── __init__.py
│   └── production_evaluator.py
└── utils/                   # Utilities
    ├── __init__.py
    └── logger.py
```

checkpoints/                # Training checkpoints
├── checkpoint_epoch_10.pt
├── checkpoint_epoch_20.pt
└── ...

results/                    # Training results and reports
├── training_summary.json
├── evaluation_results.json
└── visualization_outputs/
```

## 🚨 Important Improvements

**✅ Enhanced Architecture**: Multi-model ensemble with LSTM, Transformer, and CNN-LSTM
**✅ Adversarial Robustness**: Built-in adversarial training and evaluation
**✅ Explainable AI**: Attention weights, feature importance, and explanations
**✅ Model Management**: Versioning, export formats (ONNX, TorchScript, quantized)
**✅ Advanced Evaluation**: Comprehensive metrics, cross-validation, stratified evaluation
**✅ Production Ready**: Optimized for low-latency inference and deployment

---

**Status**: ✅ Production Ready | **Version**: 3.0.0 | **Last Updated**: 2025-10-08
    return {"status": "healthy", "model_loaded": engine is not None}
```

### Rust Integration
```python
# Example Python service to be called from Rust via HTTP API
# Or via direct embedding using PyO3 if preferred
```

## 📚 API Reference

### Core Classes

#### `BehavioralAnalyzer`
```python
model = BehavioralAnalyzer(
    input_size=7,           # Number of input features
    hidden_size=128,        # LSTM hidden units
    num_layers=4,           # Number of LSTM layers
    dropout=0.3,            # Dropout rate
    use_attention=True,     # Whether to use attention mechanism
    bidirectional=True      # Whether to use bidirectional LSTM
)

# Forward pass
output = model(torch.tensor(input_sequence))  # Returns anomaly score (0-1)

# Get embeddings for analysis
embeddings = model.get_embeddings(torch.tensor(input_sequence))

# Predict anomaly type
anomaly_types = model.predict_anomaly_type(torch.tensor(input_sequence))
```

#### `InferenceEngine`
```python
engine = InferenceEngine(
    model=model,
    feature_extractor=feature_extractor,
    threshold=0.5,                   # Classification threshold (0-1)
    enable_explainability=True       # Enable feature explanations
)

# Single analysis
result = engine.analyze(access_sequence)  # numpy array (seq_len, features)

# Batch analysis
results = engine.batch_analyze([seq1, seq2, seq3], parallel=True)

# Get attention heatmap for visualization
attention_weights = engine.get_attention_heatmap(access_sequence)
```

#### `AnalysisResult`
```python
@dataclass
class AnalysisResult:
    access_pattern_score: float        # 0-1 anomaly score
    behavior_normal: bool              # True if behavior is normal
    anomaly_detected: bool             # True if anomaly detected
    confidence: float                  # 0-1 confidence in prediction
    last_updated: str                  # Timestamp of analysis
    threat_score: float                # 0-100 threat level
    anomaly_type: Optional[str]        # Anomaly type name (if detected)
    anomaly_type_confidence: Optional[float]  # Confidence in type classification
    attention_weights: Optional[List[float]]  # Attention weights for each timestep
    explanation: Optional[Dict]        # Feature-based explanation

# Convert to dictionary for API response
result_dict = result.to_dict()
```

#### `ModelManager`
```python
manager = ModelManager(model_dir=Path('./models'))

# Save trained model with versioning
manager.save_model(
    model=trained_model,
    version="1.0.0",
    metrics=evaluation_metrics,
    feature_extractor=feature_extractor,
    export_formats=['onnx', 'torchscript', 'quantized']
)

# Load model
model, feature_extractor = manager.load_model(version="1.0.0")

# Benchmark performance
perf_metrics = manager.benchmark_model(
    model=model,
    input_shape=(1, 20, 7),
    num_iterations=1000
)
```

#### `EnsembleManager`
```python
ensemble_mgr = EnsembleManager(
    input_size=7,
    ensemble_size=5,
    base_architecture='mixed',  # lstm, transformer, cnn-lstm, or mixed
    hidden_size=128,
    num_layers=3
)

# Train ensemble
histories = ensemble_mgr.train_ensemble(dataloaders, epochs=50)

# Make predictions
ensemble_pred, individual_preds = ensemble_mgr.predict(input_tensor)

# Evaluate ensemble
metrics = ensemble_mgr.evaluate_ensemble(test_loader)

# Update weights based on performance
ensemble_mgr.update_weights(validation_loader)
```

## 🎓 Academic Background & Research

This implementation integrates cutting-edge research in:

- **Sequential Anomaly Detection**: LSTM networks for temporal pattern recognition
- **Attention Mechanisms**: Self-attention for temporal pattern focus
- **Ensemble Learning**: Combining multiple architectures for improved robustness
- **Adversarial Training**: Improving model robustness against adversarial examples
- **Explainable AI**: Attention weights and feature importance for interpretability
- **Transformer Architectures**: For complex sequence modeling
- **CNN-LSTM Hybrids**: Combining local pattern extraction with temporal modeling

## 📈 Performance Benchmarks

### Standard Model
- **Accuracy**: 98.5%+ on realistic test sets
- **F1 Score**: 98.2%+ 
- **ROC-AUC**: 0.99+ 
- **Inference Latency**: <10ms GPU, <25ms CPU
- **Memory Usage**: ~50MB GPU, ~100MB CPU

### Ensemble Model (Recommended)
- **Accuracy**: 99.2%+
- **F1 Score**: 99.0%+
- **Robustness**: 95%+ accuracy under adversarial attacks
- **Diversity**: 0.15+ average standard deviation across models

### Adversarial Robustness
- **Clean Accuracy**: 99.0%+
- **Robust Accuracy (FGSM)**: 95%+ at ε=0.01
- **Robust Accuracy (PGD)**: 92%+ at ε=0.01

## 📝 License

MIT License - See [LICENSE](../LICENSE) for details

## 🤝 Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines

## 📧 Support

For issues or questions:
- GitHub Issues: [SentinelFS Issues](https://github.com/reicalasso/SentinelFS/issues)
- Email: support@sentinelfs.io

---

**Built with ❤️ for production security systems**

*Last Updated: October 7, 2025*
