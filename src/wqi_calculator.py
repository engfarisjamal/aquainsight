"""Water Quality Index calculator."""

import numpy as np
import pandas as pd


class WQICalculator:
    """Calculate Water Quality Index (WQI) using weighted arithmetic method."""

    # Parameters, weights, and ideal values (based on WHO standards)
    PARAMS = {
        "ph": {"weight": 0.15, "ideal": 7.0, "std": 0.5, "unit": ""},
        "dissolved_oxygen": {"weight": 0.25, "ideal": 8.0, "std": 1.5, "unit": "mg/L"},
        "turbidity": {"weight": 0.20, "ideal": 2.0, "std": 3.0, "unit": "NTU"},
        "nitrate": {"weight": 0.15, "ideal": 2.0, "std": 3.0, "unit": "mg/L"},
        "phosphate": {"weight": 0.10, "ideal": 0.5, "std": 0.5, "unit": "mg/L"},
        "temperature": {"weight": 0.15, "ideal": 20.0, "std": 5.0, "unit": "°C"},
    }

    def _quality_rating(self, value: float, ideal: float, std: float) -> float:
        """Return quality rating (Q-value) for a parameter. 0-100 scale."""
        if std == 0:
            return 100.0
        # Gaussian curve centered at ideal value
        diff = abs(value - ideal) / std
        q = 100 * np.exp(-0.5 * (diff ** 2))
        return float(np.clip(q, 0, 100))

    def calculate_for_row(self, row: pd.Series) -> float:
        """Calculate WQI for a single row."""
        total_weight = 0.0
        weighted_sum = 0.0

        for param, config in self.PARAMS.items():
            if param not in row or pd.isna(row[param]):
                continue
            q = self._quality_rating(float(row[param]), config["ideal"], config["std"])
            weighted_sum += config["weight"] * q
            total_weight += config["weight"]

        if total_weight == 0:
            return 0.0
        return round(weighted_sum / total_weight, 2)

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add WQI column to DataFrame."""
        df = df.copy()
        df["wqi"] = df.apply(self.calculate_for_row, axis=1)
        df["category"] = df["wqi"].apply(self.classify)
        df["status"] = df["category"].apply(self.status_emoji)
        return df

    @staticmethod
    def classify(wqi: float) -> str:
        """Classify WQI score into category."""
        if wqi >= 90:
            return "Excellent"
        elif wqi >= 70:
            return "Good"
        elif wqi >= 50:
            return "Fair"
        elif wqi >= 25:
            return "Poor"
        else:
            return "Very Poor"

    @staticmethod
    def status_emoji(category: str) -> str:
        """Map category to emoji."""
        return {
            "Excellent": "🟢",
            "Good": "🟡",
            "Fair": "🟠",
            "Poor": "🔴",
            "Very Poor": "⚫",
        }.get(category, "⚪")

    def get_alerts(self, df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
        """Get rows with WQI below threshold."""
        return df[df["wqi"] < threshold].copy()
