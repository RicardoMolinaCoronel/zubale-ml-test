# Mini-Prod ML Challenge — Churn Prediction

This repository implements a **production-ready ML pipeline** for customer churn prediction.  
It covers **training, serving, monitoring, and lightweight MLOps practices** using Python, FastAPI, Docker, and GitHub Actions.

---

## 📂 Project Structure

```
├── src/                # Core ML code (train, app, drift, agent_monitor, features, models)
├── tests/              # Pytest tests for training & inference
├── data/               # Input CSVs 
├── artifacts/          # Outputs (metrics.json, drift_report.json, agent_plan.yaml, models)
├── docker/Dockerfile   # Container spec
├── .github/workflows/ci.yml   # GitHub Actions CI
├── requirements.txt
├── design_gcp.md       # GCP deployment design
└── README.md
```

---

## 🚀 Quickstart

### 1. Install
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python -m src.train --data data/customer_churn_synth.csv --outdir artifacts
```
Produces in `artifacts/`:
- `model.pkl` — trained classifier
- `feature_pipeline.pkl` — preprocessing pipeline
- `metrics.json` — ROC-AUC, PR-AUC, Accuracy, metadata
- `feature_importances.csv` — feature importances 
- `drift_report.json` — drift metrics
- `agent_plan.yml` — agent monitor plan
Acceptance criterion: **ROC-AUC ≥ 0.83**.

### 3. Serve the API
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health` → returns `{"status": "ok"}`
- `POST /predict` → accepts a JSON list of rows (validated with Pydantic), returns probabilities and class labels.  
  - Returns **400 with helpful error messages** if fields are missing or categories are invalid.

### 4. Detect Drift
```bash
python -m src.drift --ref data/churn_ref_sample.csv --new data/churn_shifted_sample.csv
```
Generates `artifacts/drift_report.json` with **PSI** and **KS statistics**.

### 5. Agentic Monitor
```bash
python -m src.agent_monitor --metrics data/metrics_history.jsonl --drift data/drift_latest.json --out artifacts/agent_plan.yaml
```
Outputs `agent_plan.yaml` 

---

## 🧪 Tests
```bash
pytest -q
```
Covers:
- **Training**: artifacts exist, ROC-AUC ≥ 0.83  
- **Inference**: API returns probabilities ∈ [0,1] and valid class labels

---

## 🐳 Docker

Build image:
```bash
docker build -t churn-ml -f docker/Dockerfile .
```

Run training inside container:
```bash
docker run --rm churn-ml \
  python -m src.train --data data/customer_churn_synth.csv --outdir artifacts
```

Run API:
```bash
docker run --rm -p 8000:8000 churn-ml
```
# API Endpoints

This service exposes a FastAPI application with two endpoints: `/health` and `/predict`.

---

## **GET /health**

### Request
```bash
curl http://localhost:8000/health
```

### Response
```json
{
  "status": "ok"
}
```

Purpose: quick liveness check to ensure the API is running.

---

## **POST /predict**

### Request
Accepts a JSON body with a list of rows under the key `rows`.  
Each row must include all categorical and numeric features defined in the pipeline.

#### Example
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "rows": [
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
            "downtime_hours_30d": 0.0
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
            "downtime_hours_30d": 0.0
          }
        ]
      }'
```

### Response
```json
{
  "prob": [0.12, 0.85],
  "cls": [0, 1]
}
```

- **prob** → probability of churn for each row.  
- **cls** → predicted class: 0 = not churned, 1 = churned.

### Error Handling
- If a required field is missing or contains an unknown category, the API responds with HTTP **400 Bad Request**:

```json
{
  "detail": "Invalid request payload",
  "errors": [
    {
      "loc": ["rows", 0, "plan_type"],
      "msg": "value is not a valid enumeration member",
      "type": "type_error.enum"
    }
  ]
}
```


---

## ⚙️ CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push/PR

---

## ☁️ GCP Deployment

See [`design_gcp.md`](design_gcp.md) for a 1 page architecture:


---

## ✅ Deliverables

- `src/` → code (train, app, drift, monitor)  
- `tests/` → training and inference tests  
- `docker/Dockerfile` → container spec  
- `.github/workflows/ci.yml` → CI pipeline  
- `requirements.txt` → dependencies  
- `artifacts/` → metrics.json, drift_report.json, agent_plan.yaml  
- `design_gcp.md` → GCP deployment design  
- `README.md` → quickstart and documentation  

---

## 🧾 Notes
- Deterministic seed (`42`) ensures reproducible training.  
  
