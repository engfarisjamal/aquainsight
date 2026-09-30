"""Anomaly detection for water quality using Isolation Forest."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """Detect anomalies in water quality data using Isolation Forest."""

    FEATURES = ["ph", "dissolved_oxygen", "turbidity", "temperature", "nitrate", "phosphate"]

    def __init__(self, contamination: float = 0.05, random_state: int = 42) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        self.scaler = StandardScaler()

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit model and detect anomalies."""
        df = df.copy()
        X = df[self.FEATURES].fillna(df[self.FEATURES].mean())
        X_scaled = self.scaler.fit_transform(X)

        predictions = self.model.fit_predict(X_scaled)
        scores = self.model.decision_function(X_scaled)

        df["is_anomaly"] = predictions == -1
        df["anomaly_score"] = np.round(scores, 4)
        df["anomaly_label"] = df["is_anomaly"].map({True: "⚠️ Anomaly", False: "✅ Normal"})

        return df

    def get_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get only anomalous rows."""
        return df[df["is_anomaly"]].copy()

    def summary(self, df: pd.DataFrame) -> dict:
        """Summary statistics of anomalies."""
        total = len(df)
        anomalies = df["is_anomaly"].sum()
        return {
            "total_records": total,
            "anomalies_count": int(anomalies),
            "anomaly_rate": round(anomalies / total * 100, 2) if total > 0 else 0.0,
            "normal_count": int(total - anomalies),
        }

    def per_station(self, df: pd.DataFrame) -> pd.DataFrame:
        """Anomalies per station."""
        return df.groupby("station_id").agg(
            total=("is_anomaly", "count"),
            anomalies=("is_anomaly", "sum"),
        ).assign(
            rate=lambda x: (x["anomalies"] / x["total"] * 100).round(2)
        )
