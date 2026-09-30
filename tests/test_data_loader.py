"""Tests for Data Loader."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest

from data_loader import WaterQualityDataLoader


@pytest.fixture
def sample_csv(tmp_path):
    """Create sample CSV."""
    df = pd.DataFrame({
        "station_id": ["A", "A", "B"],
        "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
        "ph": [7.0, 7.2, 6.8],
        "dissolved_oxygen": [8.0, 7.5, 6.5],
        "turbidity": [2.0, 2.5, 5.0],
        "temperature": [20.0, 21.0, 19.0],
        "nitrate": [2.0, 2.5, 3.0],
        "phosphate": [0.5, 0.6, 0.7],
    })
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_load_data(sample_csv):
    """Test loading data."""
    loader = WaterQualityDataLoader(sample_csv)
    df = loader.load()
    assert len(df) == 3
    assert "date" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_get_station_data(sample_csv):
    """Test filtering by station."""
    loader = WaterQualityDataLoader(sample_csv)
    df = loader.load()
    station_a = loader.get_station_data(df, "A")
    assert len(station_a) == 2
    assert all(station_a["station_id"] == "A")


def test_missing_file():
    """Test missing file raises error."""
    loader = WaterQualityDataLoader("nonexistent.csv")
    with pytest.raises(FileNotFoundError):
        loader.load()
