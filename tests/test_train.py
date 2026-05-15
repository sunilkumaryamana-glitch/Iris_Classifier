"""Basic checks that training runs and produces expected artifacts."""

from pathlib import Path

import joblib
import pytest
from sklearn.datasets import load_iris

import train


def test_main_runs_and_writes_outputs(tmp_path, monkeypatch):
    """Run training with outputs redirected to a temp directory."""
    monkeypatch.setattr(train, "OUTPUT_DIR", tmp_path)
    train.main()
    assert (tmp_path / "model.joblib").is_file()
    assert (tmp_path / "confusion_matrix.png").is_file()


def test_saved_model_predicts_iris():
    """Smoke test: default outputs (after a local run) or skip if missing."""
    model_path = Path(train.PROJECT_ROOT) / "outputs" / "model.joblib"
    if not model_path.is_file():
        pytest.skip("Run src/train.py once to generate outputs/model.joblib")
    model = joblib.load(model_path)
    iris = load_iris()
    preds = model.predict(iris.data[:3])
    assert len(preds) == 3
    assert set(preds).issubset({0, 1, 2})
