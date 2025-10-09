import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pytest
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sentinelzer0.training import RealWorldTrainer


class DummyFeatureExtractor:
    """Minimal feature extractor that returns precomputed vectors from events."""

    def extract_from_event(self, event):
        return np.array(event["features"], dtype=np.float32)


class TinyDetector(nn.Module):
    """Lightweight detector with deterministic behaviour for testing."""

    def __init__(self, input_size: int = 30):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        # Strongly weight the final feature to control outputs deterministically.
        self.linear.weight.data[0, -1] = 5.0
        self.linear.bias.data.fill_(-2.5)

    def forward(self, x: torch.Tensor, return_components: bool = False):
        pooled = x.mean(dim=1)
        logits = self.linear(pooled)
        score = torch.sigmoid(logits)
        if return_components:
            batch, seq_len, _ = x.shape
            attention = torch.ones(batch, seq_len, 1, device=x.device) / seq_len
            components = {
                "dl_score": score,
                "if_score": score,
                "heuristic_score": score,
                "attention_weights": attention,
                "weights": {"dl": 1.0, "if": 0.0, "heuristic": 0.0},
            }
            return score, components
        return score, None

    def fit_isolation_forest(self, X):
        return

    def calibrate_thresholds(self, X, y):
        return

    def save_components(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)


@pytest.fixture
def trainer(tmp_path):
    model = TinyDetector()
    extractor = DummyFeatureExtractor()
    return RealWorldTrainer(
        model=model,
        feature_extractor=extractor,
        learning_rate=0.01,
        checkpoint_dir=str(tmp_path / "ckpts"),
        balance_classes=True,
        dynamic_class_weighting=True,
    )


def test_weighted_sampler_applied_for_imbalanced_data(trainer):
    torch.manual_seed(0)
    np.random.seed(0)
    features = np.random.rand(100, 4, 30).astype(np.float32)
    labels = np.concatenate([np.zeros(96), np.ones(4)]).astype(np.float32).reshape(-1, 1)

    loader = trainer._create_dataloader(features, labels, batch_size=16, shuffle=True)

    assert isinstance(loader.sampler, torch.utils.data.WeightedRandomSampler)
    sampler_info = trainer.metrics["sampler"]
    assert sampler_info["applied"] is True
    assert sampler_info["positive_fraction"] < trainer.min_positive_fraction


def test_find_optimal_threshold_maximizes_f1(trainer):
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    scores = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float32)

    threshold, metrics = trainer._find_optimal_threshold(y_true, scores)

    assert 0.4 < threshold < 0.9
    assert metrics["f1"] == pytest.approx(1.0, rel=1e-3)
    assert metrics["precision"] == pytest.approx(1.0, rel=1e-3)
    assert metrics["recall"] == pytest.approx(1.0, rel=1e-3)
    assert metrics["threshold"] == pytest.approx(threshold)


def test_incremental_update_returns_rich_summary(tmp_path):
    model = TinyDetector()
    extractor = DummyFeatureExtractor()
    trainer = RealWorldTrainer(
        model=model,
        feature_extractor=extractor,
        learning_rate=0.01,
        checkpoint_dir=str(tmp_path / "ckpts"),
        balance_classes=True,
        dynamic_class_weighting=True,
    )

    sequence_length = 3
    events = []
    labels = []
    for i in range(12):
        label = 1 if i % 4 == 0 else 0
        feature_vector = np.zeros(30, dtype=np.float32)
        feature_vector[0] = i / 12.0
        feature_vector[-1] = float(label)
        events.append({"features": feature_vector})
        labels.append(label)

    result = trainer.incremental_update(
        new_events=events,
        new_labels=np.array(labels, dtype=np.float32),
        num_epochs=2,
        batch_size=4,
        sequence_length=sequence_length,
    )

    assert result["num_samples"] > 0
    assert "training_metrics" in result and "evaluation_metrics" in result
    assert "avg_f1" in result["training_metrics"]
    assert "optimal_threshold" in result["evaluation_metrics"]
    assert result["threshold_updated"] is True
    assert trainer.metrics["incremental"]["evaluation"]["optimal_threshold"] == pytest.approx(
        result["evaluation_metrics"]["optimal_threshold"]
    )