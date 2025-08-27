# TODO: Pydantic schemas for /predict

# src/io_schemas.py
from typing import List, Literal
from pydantic import BaseModel, field_validator

class RowIn(BaseModel):
    plan_type: Literal["Basic","Standard","Pro"]
    contract_type: Literal["Monthly","Annual"]
    autopay: Literal["Yes","No"]
    is_promo_user: Literal["Yes","No"]
    add_on_count: float
    tenure_months: float
    monthly_usage_gb: float
    avg_latency_ms: float
    support_tickets_30d: float
    discount_pct: float
    payment_failures_90d: float
    downtime_hours_30d: float

    @field_validator("discount_pct")
    @classmethod
    def pct_bounds(cls, v):
        if v < 0 or v > 100:
            raise ValueError("discount_pct must be in [0, 100]")
        return v

class PredictRequest(BaseModel):
    rows: List[RowIn]

class PredictResponse(BaseModel):
    prob: List[float]
    cls: List[int]
