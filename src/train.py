# TODO: Implement training script.
# CLI: python -m src.train --data data/customer_churn_synth.csv --outdir artifacts/

import argparse, os, json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV


from .features import build_preprocessor, TARGET, SEED, \
    CATEGORICAL_COLS, NUMERIC_COLS
from .models import build_model
from .metrics import compute_metrics, save_json, get_git_sha

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="xgb", choices=["xgb","hgb","logreg"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data)
    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df[TARGET].astype(int)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    pre = build_preprocessor()
    X_trp = pre.fit_transform(X_tr)
    X_valp = pre.transform(X_val)

    model = build_model(args.model)

    # Always do randomized HPO (deterministic)
    HPO_TRIALS = 15
    model = randomized_hpo(model, X_trp, y_tr, HPO_TRIALS, seed=SEED)

    #model.fit(X_trp, y_tr)

    if hasattr(model, "predict_proba"):
        p_val = model.predict_proba(X_valp)[:, 1]
    else:
        s = model.decision_function(X_valp).reshape(-1, 1)
        p_val = MinMaxScaler().fit_transform(s).ravel()

    y_pred = (p_val >= 0.5).astype(int)
    m = compute_metrics(y_val, p_val, y_pred)

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "model_type": model.__class__.__name__,
    }
    save_json(os.path.join(args.outdir, "metrics.json"), {**m, **meta})

    dump(pre, os.path.join(args.outdir, "feature_pipeline.pkl"))
    dump(model, os.path.join(args.outdir, "model.pkl"))

    # Features importance
    try:
        if hasattr(model, "feature_importances_"):
            # build names
            from .features import get_feature_names
            names = get_feature_names(pre)
            pd.DataFrame({
                "feature": names[:len(model.feature_importances_)],
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)\
             .to_csv(os.path.join(args.outdir, "feature_importances.csv"), index=False)
    except Exception:
        pass

    print(json.dumps(m, indent=2))

def randomized_hpo(model, X, y, trials: int, seed: int = 42):
    """
    Run trial randomized HPO.
    Falls back to fitting the provided model if search fails.
    """
    try:
        from xgboost import XGBClassifier  # optional
        is_xgb = isinstance(model, XGBClassifier)
    except Exception:
        is_xgb = False

    if is_xgb:
        param_dist = {
            "n_estimators": [200, 300, 400, 600],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_lambda": [0.0, 0.5, 1.0, 2.0],
        }
        scoring = "roc_auc"
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        if isinstance(model, HistGradientBoostingClassifier):
            param_dist = {
                "max_leaf_nodes": [15, 31, 63],
                "learning_rate": [0.03, 0.05, 0.08, 0.1],
                "l2_regularization": [0.0, 0.5, 1.0],
            }
        else:
            model.fit(X, y)
            return model
        scoring = "roc_auc"

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=trials,
        scoring=scoring,
        cv=cv,
        random_state=seed,
        n_jobs=1,
        verbose=0,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_


if __name__ == "__main__":
    main()
