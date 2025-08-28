# TODO: Implement PSI/KS drift calc.
# CLI: python -m src.drift --ref data/churn_ref_sample.csv --new data/churn_shifted_sample.csv

import argparse, json, os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from .features import CATEGORICAL_COLS, NUMERIC_COLS

def psi_numeric(ref: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
    ref = ref[~np.isnan(ref)]
    new = new[~np.isnan(new)]
    if len(ref) == 0 or len(new) == 0:
        return float("nan")
    qs = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(ref, qs))
    if len(cuts) < 3:
        cuts = np.unique(np.concatenate([cuts, [cuts[-1] + 1e-9]]))
    r_hist, _ = np.histogram(ref, bins=cuts)
    n_hist, _ = np.histogram(new, bins=cuts)
    r_perc = r_hist / max(r_hist.sum(), 1)
    n_perc = n_hist / max(n_hist.sum(), 1)
    eps = 1e-6
    return float(np.sum((r_perc - n_perc) * np.log((r_perc + eps)/(n_perc + eps))))

def psi_categorical(ref: pd.Series, new: pd.Series) -> float:
    rc = ref.value_counts(dropna=False)
    nc = new.value_counts(dropna=False)
    cats = sorted(set(rc.index).union(nc.index))
    r = np.array([rc.get(c, 0) for c in cats], float)
    n = np.array([nc.get(c, 0) for c in cats], float)
    r = r / max(r.sum(), 1.0)
    n = n / max(n.sum(), 1.0)
    eps = 1e-6
    return float(np.sum((r - n) * np.log((r + eps)/(n + eps))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--threshold", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ref = pd.read_csv(args.ref)
    new = pd.read_csv(args.new)

    feats = {}
    for c in NUMERIC_COLS:
        psi = psi_numeric(ref[c].to_numpy(), new[c].to_numpy())
        ks = ks_2samp(ref[c].to_numpy(), new[c].to_numpy()).statistic
        feats[c] = {"psi": psi, "ks": float(ks)}
    for c in CATEGORICAL_COLS:
        feats[c] = {"psi": psi_categorical(ref[c], new[c]), "ks": None}

    overall = any(v["psi"] is not None and v["psi"] >= args.threshold for v in feats.values())
    out = {
        "threshold": args.threshold,
        "overall_drift": bool(overall),
        "features": {k: round(v["psi"], 4) if v["psi"] is not None else None for k, v in feats.items()},
        "ks": {k: (round(v["ks"], 4) if isinstance(v["ks"], float) else None) for k, v in feats.items()},
    }
    with open(os.path.join(args.outdir, "drift_report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
