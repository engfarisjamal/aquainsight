"""Tests for WQI Calculator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest

from wqi_calculator import WQICalculator


@pytest.fixture
def calculator():
    return WQICalculator()


def test_perfect_water(calculator):
    """Perfect water should have WQI close to 100."""
    row = pd.Series({
        "ph": 7.0,
        "dissolved_oxygen": 8.0,
        "turbidity": 2.0,
        "nitrate": 2.0,
        "phosphate": 0.5,
        "temperature": 20.0,
    })
    wqi = calculator.calculate_for_row(row)
    assert wqi > 95


def test_poor_water(calculator):
    """Poor water should have low WQI."""
    row = pd.Series({
        "ph": 3.0,
        "dissolved_oxygen": 1.0,
        "turbidity": 30.0,
        "nitrate": 12.0,
        "phosphate": 2.5,
        "temperature": 40.0,
    })
    wqi = calculator.calculate_for_row(row)
    assert wqi < 30


def test_classify(calculator):
    """Test WQI classification."""
    assert calculator.classify(95) == "Excellent"
    assert calculator.classify(75) == "Good"
    assert calculator.classify(60) == "Fair"
    assert calculator.classify(30) == "Poor"
    assert calculator.classify(10) == "Very Poor"


def test_status_emoji(calculator):
    """Test emoji mapping."""
    assert calculator.status_emoji("Excellent") == "🟢"
    assert calculator.status_emoji("Poor") == "🔴"


def test_calculate_dataframe(calculator):
    """Test WQI calculation on DataFrame."""
    df = pd.DataFrame({
        "ph": [7.0, 6.5],
        "dissolved_oxygen": [8.0, 6.0],
        "turbidity": [2.0, 5.0],
        "nitrate": [2.0, 3.0],
        "phosphate": [0.5, 0.8],
        "temperature": [20.0, 22.0],
    })
    result = calculator.calculate(df)
    assert "wqi" in result.columns
    assert "category" in result.columns
    assert len(result) == 2
