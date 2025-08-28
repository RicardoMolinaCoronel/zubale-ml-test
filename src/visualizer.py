# viz_help.py
# Quick helper to visualize feature importances and drift (PSI).
# Usage: python viz_help.py

import os, json
import pandas as pd
import matplotlib.pyplot as plt

# ---- Config (edit paths if needed) ----
IMP_CSV = "../artifacts/feature_importances.csv"
DRIFT_JSON = "../artifacts/drift_report.json"
TOP_N_IMPORTANCES = 15
SAVE_PNG = False  # set True to save figures as PNGs next to the files

def show_feature_importances(path_csv: str, top_n: int = 15):
    if not os.path.exists(path_csv):
        print(f"[warn] feature importances not found: {path_csv}")
        return
    df = pd.read_csv(path_csv)
    if "feature" not in df.columns or "importance" not in df.columns:
        print(f"[warn] expected columns ['feature','importance'] in {path_csv}")
        return

    df = df.sort_values("importance", ascending=False).head(top_n)
    plt.figure()
    plt.barh(df["feature"][::-1], df["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {min(top_n, len(df))} Feature Importances")
    plt.tight_layout()
    if SAVE_PNG:
        out = os.path.splitext(path_csv)[0] + "_top_importances.png"
        plt.savefig(out, dpi=150)
        print(f"[info] saved {out}")
    plt.show()

def show_drift_report(path_json: str):
    if not os.path.exists(path_json):
        print(f"[warn] drift report not found: {path_json}")
        return
    with open(path_json, "r", encoding="utf-8") as f:
        rpt = json.load(f)

    features = rpt.get("features", {}) or {}
    if not features:
        print(f"[warn] no 'features' PSI map found in {path_json}")
        return
    threshold = rpt.get("threshold", 0.2)

    drift_df = pd.DataFrame([
        {"feature": k, "psi": v} for k, v in features.items()
        if isinstance(v, (int, float))
    ]).sort_values("psi", ascending=False)

    plt.figure()
    plt.barh(drift_df["feature"][::-1], drift_df["psi"][::-1])
    plt.axvline(threshold, linestyle="--")
    plt.xlabel("PSI")
    plt.title(f"Drift (PSI) by Feature  |  threshold={threshold}")
    plt.tight_layout()
    if SAVE_PNG:
        out = os.path.splitext(path_json)[0] + "_psi.png"
        plt.savefig(out, dpi=150)
        print(f"[info] saved {out}")
    plt.show()

if __name__ == "__main__":
    show_feature_importances(IMP_CSV, TOP_N_IMPORTANCES)
    show_drift_report(DRIFT_JSON)
