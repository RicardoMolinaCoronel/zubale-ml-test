# GCP Design (Candidate to fill)

Describe BigQuery + Vertex/Cloud Run + Monitoring.



---


# ☁️ GCP Deployment Design

This document outlines how to run the churn prediction pipeline on Google Cloud Platform.

---

## 🗄️ Data Layer
- **BigQuery (BQ)** stores historical churn tables.  
- Raw CSVs ingested into staging BQ, then partitioned/clustered tables for training & drift checks.  
- Low cost (<$10/mo for GB-scale data).

---

## 🧠 Training
- **Vertex AI Training** jobs with Python 3.11 custom containers.  
- Jobs pull training data from BQ → preprocess → train → push artifacts (`model.pkl`, `pipeline.pkl`, `metrics.json`) to **Cloud Storage (GCS)**.  
- Deterministic seeds + pinned requirements ensure reproducibility.  
- Cost: ad-hoc; pay only for compute hours (~$0.15–$0.50/hr for small CPU jobs).

---

## ⚡ Serving
- Package FastAPI app as container.  
- Deploy on **Cloud Run**:
  - Autoscaling to zero when idle.
  - 1–2 vCPU, 512–1024 MB RAM.
  - HTTPS endpoint out of the box.
- Alternative: **Vertex AI Endpoints** for managed model serving (if latency/scale needs grow).

---

## 📈 Monitoring & Drift
- **Cloud Monitoring** for:
  - Latency (p95) dashboards & alerts.
  - Error rates.
- **Grafana** (optional) for richer dashboards.  
- **Drift jobs**: daily **Cloud Run Jobs** pulling random BQ samples → `drift_report.json` → GCS.  

---

## 🧑‍✈️ Agentic Monitor
- **Cloud Run Job** runs `agent_monitor.py` daily:
  - Loads `metrics_history.jsonl` and `drift_latest.json` from GCS.  
  - Emits `agent_plan.yaml` with `healthy | warn | critical`.  
  - Can trigger Cloud Build or Slack webhook for incidents.

---

## 🔐 Security
- **Secret Manager** stores DB/API credentials.  
- Service Accounts follow least-privilege principle.  
- No secrets in code or CI.

---

## 💰 Cost Notes
- **Cloud Run**: a few $/month (scale-to-zero).  
- **BigQuery**: minimal for <10GB datasets.  
- **Vertex AI Training**: billed per-job; small jobs ≈ a few $ each.  
- **Monitoring**: included free tier covers basics.

---

## ✅ Summary
- Data → BigQuery  
- Training → Vertex AI, artifacts in GCS  
- Serving → Cloud Run (containerized FastAPI)  
- Monitoring → Cloud Monitoring (+Grafana)  
- Drift/Agent → Cloud Run Jobs  
- Secure, reproducible, cost-efficient.

