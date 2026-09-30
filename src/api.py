"""FastAPI REST API for AquaInsight."""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from anomaly_detector import AnomalyDetector
from data_loader import WaterQualityDataLoader
from wqi_calculator import WQICalculator

app = FastAPI(
    title="AquaInsight API",
    description="AI-Powered Freshwater Quality Monitoring API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====== Models ======
class WaterSample(BaseModel):
    ph: float
    dissolved_oxygen: float
    turbidity: float
    temperature: float
    nitrate: float
    phosphate: float


class WQIResponse(BaseModel):
    wqi: float
    category: str
    status: str


# ====== Cache ======
_df_cache: Any = None


def get_data():
    global _df_cache
    if _df_cache is None:
        loader = WaterQualityDataLoader("data/water_quality.csv")
        df = loader.load()
        df = WQICalculator().calculate(df)
        df = AnomalyDetector(contamination=0.05).fit_predict(df)
        _df_cache = df
    return _df_cache


# ====== Routes ======
@app.get("/")
def root() -> dict:
    return {
        "name": "AquaInsight API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/stations",
            "/stations/{station_id}/summary",
            "/stats",
            "/alerts",
            "/analyze",
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "service": "aquainsight"}


@app.get("/stations")
def stations() -> dict:
    df = get_data()
    return {
        "count": int(df["station_id"].nunique()),
        "stations": sorted(df["station_id"].unique().tolist()),
    }


@app.get("/stations/{station_id}/summary")
def station_summary(station_id: str) -> dict:
    df = get_data()
    station_df = df[df["station_id"] == station_id]
    if len(station_df) == 0:
        raise HTTPException(status_code=404, detail=f"Station '{station_id}' not found")

    return {
        "station_id": station_id,
        "records": len(station_df),
        "mean_wqi": round(float(station_df["wqi"].mean()), 2),
        "min_wqi": round(float(station_df["wqi"].min()), 2),
        "max_wqi": round(float(station_df["wqi"].max()), 2),
        "anomalies": int(station_df["is_anomaly"].sum()),
        "mean_ph": round(float(station_df["ph"].mean()), 2),
        "mean_dissolved_oxygen": round(float(station_df["dissolved_oxygen"].mean()), 2),
        "mean_turbidity": round(float(station_df["turbidity"].mean()), 2),
    }


@app.get("/stats")
def stats() -> dict:
    df = get_data()
    return {
        "total_records": len(df),
        "stations": int(df["station_id"].nunique()),
        "mean_wqi": round(float(df["wqi"].mean()), 2),
        "anomalies": int(df["is_anomaly"].sum()),
        "categories": df["category"].value_counts().to_dict(),
    }


@app.get("/alerts")
def alerts(threshold: float = 50.0, limit: int = 50) -> dict:
    df = get_data()
    alerts_df = df[df["wqi"] < threshold].head(limit)
    return {
        "count": len(alerts_df),
        "threshold": threshold,
        "alerts": alerts_df[
            ["date", "station_id", "wqi", "category", "ph", "dissolved_oxygen"]
        ].to_dict(orient="records"),
    }


@app.post("/analyze", response_model=WQIResponse)
def analyze(sample: WaterSample) -> WQIResponse:
    import pandas as pd
    calc = WQICalculator()
    row = pd.Series(sample.model_dump())
    wqi = calc.calculate_for_row(row)
    category = calc.classify(wqi)
    status = calc.status_emoji(category)
    return WQIResponse(wqi=wqi, category=category, status=status)
