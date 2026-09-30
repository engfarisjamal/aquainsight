"""Visualization module for AquaInsight."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class AquaVisualizer:
    """Generate interactive charts for water quality data."""

    COLORS = {
        "Excellent": "#00CC96",
        "Good": "#FFA15A",
        "Fair": "#FFA15A",
        "Poor": "#EF553B",
        "Very Poor": "#636EFA",
        "Anomaly": "#EF553B",
        "Normal": "#00CC96",
    }

    def __init__(self, output_dir: str = "reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def wqi_timeseries(self, df: pd.DataFrame) -> go.Figure:
        """WQI over time per station."""
        fig = px.line(
            df,
            x="date",
            y="wqi",
            color="station_id",
            title="<b>Water Quality Index Over Time</b>",
            labels={"wqi": "WQI", "date": "Date", "station_id": "Station"},
        )
        fig.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Good")
        fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Fair")
        fig.update_layout(
            template="plotly_white",
            height=500,
            hovermode="x unified",
        )
        return fig

    def wqi_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Distribution of WQI categories."""
        counts = df["category"].value_counts().reset_index()
        counts.columns = ["Category", "Count"]
        fig = px.pie(
            counts,
            names="Category",
            values="Count",
            title="<b>WQI Category Distribution</b>",
            color="Category",
            color_discrete_map=self.COLORS,
            hole=0.4,
        )
        fig.update_layout(template="plotly_white", height=400)
        return fig

    def parameter_box(self, df: pd.DataFrame, param: str) -> go.Figure:
        """Box plot of a parameter per station."""
        fig = px.box(
            df,
            x="station_id",
            y=param,
            color="station_id",
            title=f"<b>{param.replace('_', ' ').title()} by Station</b>",
            labels={"station_id": "Station", param: param},
        )
        fig.update_layout(template="plotly_white", height=400, showlegend=False)
        return fig

    def anomaly_scatter(self, df: pd.DataFrame) -> go.Figure:
        """Scatter plot highlighting anomalies."""
        fig = px.scatter(
            df,
            x="date",
            y="wqi",
            color="anomaly_label",
            symbol="station_id",
            title="<b>Anomalies in Water Quality</b>",
            color_discrete_map={
                "⚠️ Anomaly": "#EF553B",
                "✅ Normal": "#00CC96",
            },
        )
        fig.update_layout(template="plotly_white", height=500)
        return fig

    def station_comparison(self, df: pd.DataFrame) -> go.Figure:
        """Compare stations by mean WQI."""
        means = df.groupby("station_id")["wqi"].mean().reset_index()
        means.columns = ["Station", "Mean WQI"]
        means = means.sort_values("Mean WQI", ascending=True)
        
        fig = px.bar(
            means,
            x="Mean WQI",
            y="Station",
            orientation="h",
            title="<b>Average WQI per Station</b>",
            color="Mean WQI",
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(template="plotly_white", height=400)
        return fig

    def save_html(self, fig: go.Figure, filename: str) -> str:
        """Save figure as HTML."""
        path = self.output_dir / filename
        fig.write_html(str(path))
        return str(path)
