"""Main analysis script for AquaInsight."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from anomaly_detector import AnomalyDetector
from data_loader import WaterQualityDataLoader
from visualizer import AquaVisualizer
from wqi_calculator import WQICalculator


def main() -> None:
    print("=" * 60)
    print("🌊 AquaInsight - Water Quality Analysis")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data...")
    loader = WaterQualityDataLoader("data/water_quality.csv")
    df = loader.load()
    print(f"  ✅ Loaded {len(df)} records from {df['station_id'].nunique()} stations")

    # 2. Calculate WQI
    print("\n[2/5] Calculating Water Quality Index...")
    calc = WQICalculator()
    df = calc.calculate(df)
    print(f"  ✅ Mean WQI: {df['wqi'].mean():.2f}")
    print(f"  ✅ Categories: {df['category'].value_counts().to_dict()}")

    # 3. Detect anomalies
    print("\n[3/5] Detecting anomalies (Isolation Forest)...")
    detector = AnomalyDetector(contamination=0.05)
    df = detector.fit_predict(df)
    summary = detector.summary(df)
    print(f"  ✅ Anomalies: {summary['anomalies_count']} ({summary['anomaly_rate']}%)")

    # 4. Generate alerts
    print("\n[4/5] Generating alerts...")
    alerts = calc.get_alerts(df, threshold=50.0)
    print(f"  ⚠️  {len(alerts)} alerts (WQI < 50)")

    # 5. Create visualizations
    print("\n[5/5] Generating visualizations...")
    viz = AquaVisualizer("reports")

    charts = {
        "wqi_timeseries.html": viz.wqi_timeseries(df),
        "wqi_distribution.html": viz.wqi_distribution(df),
        "anomalies.html": viz.anomaly_scatter(df),
        "station_comparison.html": viz.station_comparison(df),
        "ph_box.html": viz.parameter_box(df, "ph"),
        "do_box.html": viz.parameter_box(df, "dissolved_oxygen"),
    }

    for filename, fig in charts.items():
        path = viz.save_html(fig, filename)
        print(f"  ✅ {path}")

    # Save processed data
    df.to_csv("data/processed_data.csv", index=False)
    alerts.to_csv("data/alerts.csv", index=False)
    print("\n  ✅ Saved: data/processed_data.csv")
    print(f"  ✅ Saved: data/alerts.csv ({len(alerts)} alerts)")

    print("\n" + "=" * 60)
    print("🎉 Analysis complete!")
    print("=" * 60)
    print(f"\n📊 Stations monitored: {df['station_id'].nunique()}")
    print(f"📈 Total records: {len(df)}")
    print(f"💧 Mean WQI: {df['wqi'].mean():.2f}")
    print(f"⚠️  Anomalies: {summary['anomalies_count']}")
    print(f"🚨 Alerts: {len(alerts)}")
    print(f"\n📁 Open reports/wqi_timeseries.html to view the dashboard")


if __name__ == "__main__":
    main()
