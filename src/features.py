# TODO: Implement sklearn ColumnTransformer

# src/features.py
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

SEED = 42

CATEGORICAL_COLS: List[str] = [
    "plan_type", "contract_type", "autopay", "is_promo_user"
]
NUMERIC_COLS: List[str] = [
    "add_on_count",
    "tenure_months",
    "monthly_usage_gb",
    "avg_latency_ms",
    "support_tickets_30d",
    "discount_pct",
    "payment_failures_90d",
    "downtime_hours_30d",
]
TARGET = "churned"

ALLOWED_CATEGORIES = {
    "plan_type": ["Basic", "Standard", "Pro"],
    "contract_type": ["Monthly", "Annual"],
    "autopay": ["Yes", "No"],
    "is_promo_user": ["Yes", "No"],
}

def build_preprocessor() -> ColumnTransformer:
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    return ColumnTransformer([
        ("cat", cat, CATEGORICAL_COLS),
        ("num", num, NUMERIC_COLS),
    ], remainder="drop", verbose_feature_names_out=False)

def get_feature_names(pre: ColumnTransformer) -> List[str]:
    """
    Return feature names from a fitted ColumnTransformer.
    """
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        pass

    names: List[str] = []
    for name, trans, cols in pre.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            try:
                feats = list(trans.get_feature_names_out(cols))
            except TypeError:
                feats = list(trans.get_feature_names_out())
        else:
            feats = list(cols)
        names.extend(feats)
    return names


