# TODO: Boot API, call /predict using tests/sample.json

# tests/test_inference.py
# Boot API, call /predict using sample.json

import os
import sys
import json
import subprocess
import pytest
from fastapi.testclient import TestClient

DATA_PATH = "data/customer_churn_synth.csv"
ART_DIR = "artifacts"

# Hard-coded sample request (as provided)
SAMPLE_ROWS = [
    {
        "plan_type": "Standard",
        "contract_type": "Monthly",
        "autopay": "Yes",
        "is_promo_user": "No",
        "add_on_count": 1,
        "tenure_months": 8,
        "monthly_usage_gb": 130.5,
        "avg_latency_ms": 145.0,
        "support_tickets_30d": 0,
        "discount_pct": 10.0,
        "payment_failures_90d": 0,
        "downtime_hours_30d": 0.0,   # ensure all features are present
    },
    {
        "plan_type": "Basic",
        "contract_type": "Monthly",
        "autopay": "No",
        "is_promo_user": "Yes",
        "add_on_count": 0,
        "tenure_months": 2,
        "monthly_usage_gb": 70.0,
        "avg_latency_ms": 210.0,
        "support_tickets_30d": 3,
        "discount_pct": 0.0,
        "payment_failures_90d": 2,
        "downtime_hours_30d": 0.0,
    }
]

@pytest.mark.order(2)
def test_api_predict_returns_probs_and_classes():
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Dataset not found at {DATA_PATH}")

    # Ensure artifacts exist by running training
    subprocess.run(
        [sys.executable, "-m", "src.train", "--data", DATA_PATH, "--outdir", ART_DIR],
        check=True,
    )

    # Import app AFTER artifacts exist
    from src.app import app
    client = TestClient(app)

    resp = client.post("/predict", json={"rows": SAMPLE_ROWS})
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert "prob" in body and "cls" in body
    assert len(body["prob"]) == len(SAMPLE_ROWS)
    assert len(body["cls"]) == len(SAMPLE_ROWS)

    for p in body["prob"]:
        assert 0.0 <= p <= 1.0
    for c in body["cls"]:
        assert c in (0, 1)

