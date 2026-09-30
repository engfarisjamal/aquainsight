"""PDF Report Generator for AquaInsight."""

from datetime import datetime
from pathlib import Path

import pandas as pd
from fpdf import FPDF


class AquaReportGenerator:
    """Generate professional PDF reports."""

    def __init__(self, output_dir: str = "reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate(self, df: pd.DataFrame, alerts: pd.DataFrame, filename: str = "aquainsight_report.pdf") -> str:
        """Generate PDF report."""
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # ====== Header ======
        pdf.set_fill_color(0, 204, 150)
        pdf.rect(0, 0, 210, 40, "F")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_y(12)
        pdf.cell(0, 15, "AquaInsight", ln=True, align="C")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 5, "AI-Powered Freshwater Quality Report", ln=True, align="C")

        # ====== Date ======
        pdf.set_text_color(80, 80, 80)
        pdf.set_y(45)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="R")

        # ====== Executive Summary ======
        pdf.set_y(55)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "1. Executive Summary", ln=True)

        pdf.set_font("Helvetica", "", 10)
        summary = (
            f"This report presents an analysis of freshwater quality data from "
            f"{df['station_id'].nunique()} monitoring stations over "
            f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}. "
            f"A total of {len(df):,} records were analyzed using the Weighted Arithmetic "
            f"Water Quality Index (WQI) and Isolation Forest anomaly detection."
        )
        pdf.multi_cell(0, 6, summary)
        pdf.ln(3)

        # ====== Key Metrics Table ======
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "2. Key Metrics", ln=True)

        metrics = [
            ("Total Records", f"{len(df):,}"),
            ("Monitoring Stations", str(df["station_id"].nunique())),
            ("Mean WQI", f"{df['wqi'].mean():.2f}"),
            ("Minimum WQI", f"{df['wqi'].min():.2f}"),
            ("Maximum WQI", f"{df['wqi'].max():.2f}"),
            ("Anomalies Detected", f"{int(df['is_anomaly'].sum())}"),
            ("Active Alerts (WQI < 50)", f"{len(alerts)}"),
        ]

        pdf.set_font("Helvetica", "", 10)
        pdf.set_fill_color(240, 240, 240)
        for label, value in metrics:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(90, 7, label, border=1, fill=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 7, value, border=1, ln=True)

        pdf.ln(5)

        # ====== WQI Categories ======
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "3. WQI Categories Distribution", ln=True)

        categories = df["category"].value_counts()
        pdf.set_font("Helvetica", "", 10)
        for category, count in categories.items():
            pct = count / len(df) * 100
            pdf.cell(0, 6, f"  - {category}: {count} ({pct:.1f}%)", ln=True)

        pdf.ln(5)

        # ====== Station Performance ======
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "4. Station Performance", ln=True)

        station_stats = df.groupby("station_id").agg({
            "wqi": ["mean", "min", "max"],
            "is_anomaly": "sum",
        }).round(2)

        # Table header
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(0, 204, 150)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(40, 7, "Station", border=1, fill=True, align="C")
        pdf.cell(30, 7, "Mean WQI", border=1, fill=True, align="C")
        pdf.cell(30, 7, "Min WQI", border=1, fill=True, align="C")
        pdf.cell(30, 7, "Max WQI", border=1, fill=True, align="C")
        pdf.cell(30, 7, "Anomalies", border=1, fill=True, align="C", ln=True)

        # Table rows
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 9)
        fill = False
        for station, row in station_stats.iterrows():
            if fill:
                pdf.set_fill_color(240, 240, 240)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.cell(40, 7, str(station), border=1, fill=True, align="C")
            pdf.cell(30, 7, f"{row[('wqi', 'mean')]:.2f}", border=1, fill=True, align="C")
            pdf.cell(30, 7, f"{row[('wqi', 'min')]:.2f}", border=1, fill=True, align="C")
            pdf.cell(30, 7, f"{row[('wqi', 'max')]:.2f}", border=1, fill=True, align="C")
            pdf.cell(30, 7, str(int(row[('is_anomaly', 'sum')])), border=1, fill=True, align="C", ln=True)
            fill = not fill

        pdf.ln(8)

        # ====== Alerts Section ======
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "5. Critical Alerts", ln=True)
        pdf.set_font("Helvetica", "", 10)

        if len(alerts) > 0:
            pdf.multi_cell(0, 6, f"{len(alerts)} alerts detected (WQI < 50). Top 10 listed below:")
            pdf.ln(2)

            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(239, 85, 59)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(30, 7, "Date", border=1, fill=True, align="C")
            pdf.cell(35, 7, "Station", border=1, fill=True, align="C")
            pdf.cell(25, 7, "WQI", border=1, fill=True, align="C")
            pdf.cell(25, 7, "pH", border=1, fill=True, align="C")
            pdf.cell(35, 7, "DO (mg/L)", border=1, fill=True, align="C", ln=True)

            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 9)
            for _, row in alerts.head(10).iterrows():
                pdf.cell(30, 6, row["date"].strftime("%Y-%m-%d"), border=1, align="C")
                pdf.cell(35, 6, str(row["station_id"]), border=1, align="C")
                pdf.cell(25, 6, f"{row['wqi']:.2f}", border=1, align="C")
                pdf.cell(25, 6, f"{row['ph']:.2f}", border=1, align="C")
                pdf.cell(35, 6, f"{row['dissolved_oxygen']:.2f}", border=1, align="C", ln=True)
        else:
            pdf.cell(0, 6, "No critical alerts detected.", ln=True)

        # ====== Recommendations ======
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "6. Recommendations", ln=True)

        pdf.set_font("Helvetica", "", 10)
        recommendations = [
            "Monitor stations with consistently low WQI values (< 50) on a daily basis.",
            "Investigate the root causes of detected anomalies (Industrial discharge, agricultural runoff).",
            "Implement real-time IoT sensors at high-risk stations for continuous monitoring.",
            "Establish a rapid response protocol for critical alerts to protect public health.",
            "Conduct regular water treatment audits to maintain quality standards.",
        ]
        for i, rec in enumerate(recommendations, 1):
            pdf.multi_cell(0, 6, f"{i}. {rec}")
            pdf.ln(1)

        # ====== Footer ======
        pdf.set_y(-20)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 10, "AquaInsight - OneAquaHealth Hackathon 2025", align="C")

        # ====== Save ======
        output_path = self.output_dir / filename
        pdf.output(str(output_path))
        return str(output_path)
