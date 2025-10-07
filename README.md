# SentinelFS AI – Production Behavioral Analyzer

This repository now contains only the production-ready SentinelFS AI inference stack and the reference model artifacts. Training utilities, experimental datasets, and development notebooks have been removed to deliver a lean deployment package.

## Whats Included

- `sentinelfs_ai/`: Python package with the behavioral analyzer, inference engine, model management helpers, and supporting data utilities
- `models/`: Reference checkpoints for the production ensemble (including `behavioral_analyzer_production.pt`)
- `load_model.py`: Simple script for loading the packaged model and running a quick inference demo
- `MODEL_REPORT.md`: Detailed performance report for the shipped model release (v1.0.0)
- `requirements.txt`: Minimal runtime dependencies required for inference and export workflows

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python load_model.py
```

The demo script loads the production model, analyzes randomly generated access sequences, and prints anomaly detection results. Replace the generated sample data with your own feature sequences (shape: `seq_len x 7`).

## Package Usage

```python
from sentinelfs_ai import InferenceEngine, ModelManager

model_manager = ModelManager(model_dir="models")
model, feature_extractor = model_manager.load_model(version="latest")

engine = InferenceEngine(
    model=model,
    feature_extractor=feature_extractor,
    threshold=0.5,
    enable_explainability=True,
)

analysis = engine.analyze(access_sequence)
if analysis.anomaly_detected:
    print(f"Threat score: {analysis.threat_score:.1f}")
    print(f"Type: {analysis.anomaly_type}")
```

## Additional Resources

- Full technical documentation lives in `sentinelfs_ai/README.md`
- Comprehensive metrics, confusion matrix, and deployment recommendations are in `MODEL_REPORT.md`

## Regenerating Artifacts

If you need to retrain or regenerate datasets, clone the original full repository history (prior to this cleanup) or author your own training pipeline using the modules in `sentinelfs_ai.training`.
