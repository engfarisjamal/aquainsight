"""Data loader for AquaInsight."""

from pathlib import Path

import pandas as pd


class WaterQualityDataLoader:
    """Load and preprocess water quality data."""

    REQUIRED_COLUMNS = [
        "station_id",
        "date",
        "ph",
        "dissolved_oxygen",
        "turbidity",
        "temperature",
        "nitrate",
        "phosphate",
    ]

    def __init__(self, data_path: str = "data/water_quality.csv") -> None:
        self.data_path = Path(data_path)

    def load(self) -> pd.DataFrame:
        """Load data from CSV."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        df = pd.read_csv(self.data_path)
        df["date"] = pd.to_datetime(df["date"])

        self._validate(df)
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        """Validate required columns."""
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def get_station_data(self, df: pd.DataFrame, station_id: str) -> pd.DataFrame:
        """Filter data by station."""
        return df[df["station_id"] == station_id].copy()

    def get_latest(self, df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
        """Get latest N records per station."""
        return (
            df.sort_values("date")
            .groupby("station_id")
            .tail(n)
            .reset_index(drop=True)
        )

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics per station."""
        return df.groupby("station_id").agg(
            {
                "ph": ["mean", "std", "min", "max"],
                "dissolved_oxygen": ["mean", "std"],
                "turbidity": ["mean", "std"],
                "temperature": ["mean"],
                "nitrate": ["mean"],
                "phosphate": ["mean"],
            }
        ).round(2)
