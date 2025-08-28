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
Outputs `agent_plan.yaml` with:
- `status`: `healthy | warn | critical`
- `findings`: drop %, latency, drift flag
- `actions`: e.g. `trigger_retraining`, `open_incident`
- `rationale`: rule-based explanation

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

---

## ⚙️ CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push/PR:
- Setup Python 3.11
- Install dependencies
- Run tests
- Build Docker image

---

## ☁️ GCP Deployment

See [`design_gcp.md`](design_gcp.md) for a 1 page architecture:
- Data in **BigQuery**
- Training on **Vertex AI** jobs → artifacts in **GCS**
- Serving via **Cloud Run** (containerized FastAPI)
- Monitoring in **Cloud Monitoring** / Grafana
- Drift detection and agent monitor as **Cloud Run Jobs**
- Costs: only pay-per-use, scale to zero on serving

---

## ✅ Deliverables

- `src/` → code (train, app, drift, monitor)  
- `tests/` → training & inference tests  
- `docker/Dockerfile` → container spec  
- `.github/workflows/ci.yml` → CI pipeline  
- `requirements.txt` → dependencies  
- `artifacts/` → metrics.json, drift_report.json, agent_plan.yaml  
- `design_gcp.md` → GCP deployment design  
- `README.md` → quickstart and documentation  

---

## 🧾 Notes
- Deterministic seed (`42`) ensures reproducible training.  
  
