"""Tests for Anomaly Detector."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import pytest

from anomaly_detector import AnomalyDetector


@pytest.fixture
def sample_data():
    """Create sample data with known anomalies."""
    np.random.seed(42)
    normal = pd.DataFrame({
        "ph": np.random.normal(7.0, 0.3, 100),
        "dissolved_oxygen": np.random.normal(8.0, 0.5, 100),
        "turbidity": np.random.normal(2.0, 0.5, 100),
        "temperature": np.random.normal(20.0, 2.0, 100),
        "nitrate": np.random.normal(2.0, 0.5, 100),
        "phosphate": np.random.normal(0.5, 0.1, 100),
    })
    # Add anomalies
    anomalies = pd.DataFrame({
        "ph": [2.0, 12.0],
        "dissolved_oxygen": [0.5, 15.0],
        "turbidity": [40.0, 50.0],
        "temperature": [5.0, 40.0],
        "nitrate": [15.0, 18.0],
        "phosphate": [3.0, 4.0],
    })
    return pd.concat([normal, anomalies], ignore_index=True)


def test_detector_initialization():
    """Test detector initialization."""
    detector = AnomalyDetector(contamination=0.05)
    assert detector.contamination == 0.05


def test_fit_predict(sample_data):
    """Test anomaly detection."""
    detector = AnomalyDetector(contamination=0.05)
    result = detector.fit_predict(sample_data)
    assert "is_anomaly" in result.columns
    assert "anomaly_score" in result.columns
    assert result["is_anomaly"].sum() > 0


def test_summary(sample_data):
    """Test summary statistics."""
    detector = AnomalyDetector(contamination=0.05)
    result = detector.fit_predict(sample_data)
    summary = detector.summary(result)
    assert summary["total_records"] == 102
    assert summary["anomalies_count"] > 0
    assert 0 <= summary["anomaly_rate"] <= 100
