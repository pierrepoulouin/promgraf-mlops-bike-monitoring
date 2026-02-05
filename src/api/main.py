import logging
import time
import datetime
from typing import Any, Optional

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from evidently import Report
from evidently.presets import DataDriftPreset

from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel, Field

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CollectorRegistry,
)

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------
app = FastAPI(
    title="Bike Sharing Predictor API",
    description="API for predicting bike sharing demand with MLOps monitoring.",
    version="1.0.0",
)

# -------------------------------------------------------------------
# Prometheus Metrics
# -------------------------------------------------------------------
registry = CollectorRegistry()

api_requests_total = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status_code"],
    registry=registry,
)

api_request_duration_seconds = Histogram(
    "api_request_duration_seconds",
    "API request latency",
    ["endpoint", "method", "status_code"],
    registry=registry,
)

model_rmse_score = Gauge(
    "model_rmse_score",
    "RMSE of the regression model",
    registry=registry,
)

model_mae_score = Gauge(
    "model_mae_score",
    "MAE of the regression model",
    registry=registry,
)

model_r2_score = Gauge(
    "model_r2_score",
    "R2 score of the regression model",
    registry=registry,
)

model_mape_score = Gauge(
    "model_mape_score",
    "MAPE of the regression model",
    registry=registry,
)

data_drift_detected = Gauge(
    "data_drift_detected",
    "Whether Evidently detected data drift (1=yes, 0=no)",
    registry=registry,
)

# -------------------------------------------------------------------
# Constants & Globals
# -------------------------------------------------------------------
TARGET = "cnt"

NUM_FEATS = ["temp", "atemp", "hum", "windspeed", "mnth", "hr", "weekday"]
CAT_FEATS = ["season", "holiday", "workingday", "weathersit"]

model: Pipeline | None = None
reference_data: pd.DataFrame | None = None

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return float(
        np.mean(
            np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])
        )
    )


def _fetch_reference_data() -> pd.DataFrame:
    """
    Load reference data (January 2011) bundled with the API.
    """
    return pd.read_csv("data/bike_sharing_jan_2011.csv")


def _train_reference_model(df: pd.DataFrame) -> Pipeline:
    X = df[NUM_FEATS + CAT_FEATS]
    y = df[TARGET]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_FEATS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
        ]
    )

    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", rf),
        ]
    )

    pipeline.fit(X, y)
    return pipeline

# -------------------------------------------------------------------
# Pydantic Models
# -------------------------------------------------------------------
class BikeSharingInput(BaseModel):
    temp: float
    atemp: float
    hum: float
    windspeed: float
    mnth: int
    hr: int
    weekday: int
    season: int
    holiday: int
    workingday: int
    weathersit: int
    dteday: datetime.date


class PredictionOutput(BaseModel):
    predicted_count: float


class EvaluationData(BaseModel):
    data: list[dict[str, Any]]
    evaluation_period_name: str = "unknown_period"


class EvaluationReportOutput(BaseModel):
    message: str
    rmse: Optional[float]
    mape: Optional[float]
    mae: Optional[float]
    r2score: Optional[float]
    drift_detected: int
    evaluated_items: int

# -------------------------------------------------------------------
# Prometheus Middleware
# -------------------------------------------------------------------
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response: Response | None = None

    try:
        response = await call_next(request)
        return response
    finally:
        duration = time.time() - start_time
        status_code = response.status_code if response else 500

        api_requests_total.labels(
            endpoint=request.url.path,
            method=request.method,
            status_code=status_code,
        ).inc()

        api_request_duration_seconds.labels(
            endpoint=request.url.path,
            method=request.method,
            status_code=status_code,
        ).observe(duration)

# -------------------------------------------------------------------
# Startup
# -------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    global model, reference_data
    logger.info("Loading reference data and training model...")
    reference_data = _fetch_reference_data()
    model = _train_reference_model(reference_data)
    logger.info("Model trained and ready.")

# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Bike Sharing Predictor API"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(registry),
        media_type="text/plain",
    )

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: BikeSharingInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([input_data.dict()])
    prediction = model.predict(df)[0]

    return PredictionOutput(predicted_count=float(prediction))

@app.post("/evaluate", response_model=EvaluationReportOutput)
async def evaluate(data: EvaluationData):
    if model is None or reference_data is None:
        raise HTTPException(status_code=500, detail="Model not ready")

    df_current = pd.DataFrame(data.data)

    if TARGET not in df_current.columns:
        raise HTTPException(
            status_code=400,
            detail="Missing target 'cnt' in evaluation data",
        )

    X_current = df_current[NUM_FEATS + CAT_FEATS]
    y_true = df_current[TARGET].values
    y_pred = model.predict(X_current)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = float(
        1 - np.sum((y_true - y_pred) ** 2)
        / np.sum((y_true - np.mean(y_true)) ** 2)
    )
    mape = _mape(y_true, y_pred)

    model_rmse_score.set(rmse)
    model_mae_score.set(mae)
    model_r2_score.set(r2)
    model_mape_score.set(mape)

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_data[NUM_FEATS + CAT_FEATS],
        current_data=X_current,
    )

    drift_detected = int(
        report.as_dict()["metrics"][0]["result"]["dataset_drift"]
    )
    data_drift_detected.set(drift_detected)

    return EvaluationReportOutput(
        message=f"Evaluation completed for {data.evaluation_period_name}",
        rmse=rmse,
        mae=mae,
        r2score=r2,
        mape=mape,
        drift_detected=drift_detected,
        evaluated_items=len(df_current),
    )
