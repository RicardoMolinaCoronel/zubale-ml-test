# TODO: Implement FastAPI app for churn inference.
# Endpoints: GET /health, POST /predict

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import status
from joblib import load
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from .io_schemas import PredictRequest, PredictResponse
from .features import CATEGORICAL_COLS, NUMERIC_COLS

ART = os.environ.get("ARTIFACTS_DIR", "artifacts")
PRE = os.path.join(ART, "feature_pipeline.pkl")
MODEL = os.path.join(ART, "model.pkl")

app = FastAPI(title="Churn Classifier")

_pre = None
_model = None
if os.path.exists(PRE) and os.path.exists(MODEL):
    _pre = load(PRE)
    _model = load(MODEL)

@app.get("/health")
def health():
    return {"status": "ok"}

def _ensure_ready():
    if _pre is None or _model is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Run training first.")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    _ensure_ready()
    rows = [r.model_dump() for r in req.rows]
    df = pd.DataFrame(rows, columns=CATEGORICAL_COLS + NUMERIC_COLS)
    X = _pre.transform(df)
    if hasattr(_model, "predict_proba"):
        prob = _model.predict_proba(X)[:, 1]
    else:
        s = _model.decision_function(X).reshape(-1, 1)
        prob = MinMaxScaler().fit_transform(s).ravel()
    cls = (prob >= 0.5).astype(int).tolist()
    return {"prob": [float(p) for p in prob], "cls": cls}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": "Invalid request payload",
            "errors": exc.errors(),
        },
    )