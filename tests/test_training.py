# TODO: Train, assert artifacts exist and ROC-AUC threshold
# tests/test_training.py
# Train, assert artifacts exist and ROC-AUC threshold

import json
import os
import sys
import subprocess
import pytest

DATA_PATH = "data/customer_churn_synth.csv"
ART_DIR = "artifacts"

@pytest.mark.order(1)
def test_training_produces_artifacts_and_quality():
    # Skip in CI/local if dataset is missing
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Dataset not found at {DATA_PATH}")

    # Run training
    cmd = [sys.executable, "-m", "src.train", "--data", DATA_PATH, "--outdir", ART_DIR]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"Training failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    # Artifacts exist
    metrics_path = os.path.join(ART_DIR, "metrics.json")
    model_path = os.path.join(ART_DIR, "model.pkl")
    preproc_path = os.path.join(ART_DIR, "feature_pipeline.pkl")

    assert os.path.exists(metrics_path), "metrics.json not found"
    assert os.path.exists(model_path), "model.pkl not found"
    assert os.path.exists(preproc_path), "feature_pipeline.pkl not found"

    # Quality gate
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "roc_auc" in metrics, "roc_auc missing in metrics.json"
    assert metrics["roc_auc"] >= 0.83, f"ROC-AUC below acceptance: {metrics['roc_auc']}"

