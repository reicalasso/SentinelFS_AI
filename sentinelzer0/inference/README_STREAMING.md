# Real-Time Stream Processing for SentinelFS AI

This module provides real-time inference capabilities for continuous file system event analysis.

## Quick Start

```python
from sentinelzer0.models.hybrid_detector import HybridThreatDetector
from sentinelzer0.inference.streaming_engine import StreamingInferenceEngine

# Load trained model
model = HybridThreatDetector(input_size=30)
# model.load_state_dict(torch.load('model.pt'))

# Create streaming engine
engine = StreamingInferenceEngine(
    model=model,
    sequence_length=64,
    threshold=0.5,
    device='auto'
)

# Process events
event = {
    'event_type': 'MODIFY',
    'path': '/home/user/document.txt',
    'timestamp': time.time(),
    'size': 1024,
    'is_directory': False,
    'user': 'user',
    'process': 'python3',
    'extension': '.txt'
}

prediction = engine.process_event(event)
if prediction and prediction.is_threat:
    print(f"Threat detected! Score: {prediction.threat_score:.4f}")
```

## Components

### StreamBuffer
Thread-safe sliding window buffer for event streams.

**Features:**
- Fixed-size deque-based storage
- O(1) operations
- Automatic feature caching
- Thread-safe with locks

### StreamingInferenceEngine
Real-time inference engine with GPU acceleration.

**Features:**
- <1ms median latency
- 1,000+ events/sec throughput
- GPU auto-detection
- Performance tracking
- Batch processing support

### ThreatPrediction
Structured prediction result with metadata.

## Performance

- **Median Latency**: 0.86ms
- **Throughput**: 1,197 events/sec
- **GPU Support**: CUDA auto-detection
- **Thread-Safe**: Yes

## Testing

Run the comprehensive test suite:

```bash
python3 test_phase_1_1_streaming.py
```

## API Reference

See [PHASE_1_1_SUMMARY.md](../PHASE_1_1_SUMMARY.md) for complete documentation.
