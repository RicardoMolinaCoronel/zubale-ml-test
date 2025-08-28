# TODO: Train/save/load utilities

from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from .features import SEED


def build_model(kind: str = "xgb") -> Any:
    if kind == "xgb":
        try:
            from xgboost import XGBClassifier  # type: ignore
            return XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=SEED,
                n_jobs=0,
                tree_method="hist",
                eval_metric="logloss",
            )
        except Exception:
            kind = "hgb"
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=None, max_leaf_nodes=31, learning_rate=0.08,
            l2_regularization=0.0, random_state=SEED
        )
    return LogisticRegression(solver="lbfgs", max_iter=2000, n_jobs=1, random_state=SEED)
