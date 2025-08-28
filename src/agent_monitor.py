# TODO: Implement Agentic Monitor (LLM-optional)
# CLI: python -m src.agent_monitor --metrics data/metrics_history.jsonl --drift data/drift_latest.json --out artifacts/agent_plan.yaml

import argparse, json, os, statistics, yaml
from typing import List, Dict, Any

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out

def median_last_7(vals: List[float]) -> float:
    return statistics.median(vals[-7:]) if vals else float("nan")

def build_plan(history: List[Dict[str, Any]], drift: Dict[str, Any]) -> Dict[str, Any]:
    # Extract series
    aucs = [r.get("roc_auc") for r in history if r.get("roc_auc") is not None]
    pr_aucs = [r.get("pr_auc") for r in history if r.get("pr_auc") is not None]
    lat = [r.get("latency_p95_ms") for r in history if r.get("latency_p95_ms") is not None]

    findings = []
    status = "healthy"

    # ROC-AUC drop %
    drop_auc = 0.0
    if aucs:
        med7 = median_last_7(aucs)
        latest = aucs[-1]
        if med7 and med7 > 0:
            drop_auc = (med7 - latest) / med7 * 100.0
        findings.append({"roc_auc_drop_pct": round(drop_auc, 2)})

    # PR-AUC drop %
    drop_pr = 0.0
    if pr_aucs:
        med7 = median_last_7(pr_aucs)
        latest = pr_aucs[-1]
        if med7 and med7 > 0:
            drop_pr = (med7 - latest) / med7 * 100.0

    # Latency rule
    last_two_high = False
    if len(lat) >= 2:
        last_two_high = (lat[-1] > 400) and (lat[-2] > 400)
        findings.append({"latency_p95_ms": float(lat[-1])})

    # Drift flag
    drift_overall = bool(drift.get("overall_drift")) if drift else False
    findings.append({"drift_overall": drift_overall})

    # Apply rules
    if drop_auc >= 6.0 or (drift_overall and drop_pr >= 5.0):
        status = "critical"
    elif drop_auc >= 3.0 or last_two_high:
        status = "warn"

    # Actions
    if status == "critical":
        actions = ["open_incident", "trigger_retraining", "roll_back_model", "page_oncall=false"]
    elif status == "warn":
        actions = ["trigger_retraining", "raise_thresholds", "page_oncall=false"]
    else:
        actions = ["do_nothing"]

    return {
        "status": status,
        "findings": findings,
        "actions": actions,
        "rationale": (
            "Heuristics: warn if AUC drop ≥3% vs 7-day median or p95 latency >400ms for two points; "
            "critical if AUC drop ≥6% or (overall_drift true AND PR-AUC drop ≥5%)."
        )
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--drift", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hist = load_jsonl(args.metrics)
    try:
        with open(args.drift, "r", encoding="utf-8") as f:
            drift = json.load(f)
    except Exception:
        drift = {}

    plan = build_plan(hist, drift)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(plan, f, sort_keys=False, allow_unicode=True)
    print(yaml.safe_dump(plan, sort_keys=False, allow_unicode=True))

if __name__ == "__main__":
    main()

