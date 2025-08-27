# TODO: Compute ROC-AUC, PR-AUC, accuracy

# src/metrics.py
import json, os, subprocess
from typing import Dict, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

def compute_metrics(y_true, y_prob, y_pred) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def get_git_sha() -> Optional[str]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return sha
    except Exception:
        return None
