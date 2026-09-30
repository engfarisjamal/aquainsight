"""AquaInsight - Interactive Water Quality Dashboard."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from anomaly_detector import AnomalyDetector
from data_loader import WaterQualityDataLoader
from wqi_calculator import WQICalculator

# ====== Page Config ======
st.set_page_config(
    page_title="AquaInsight",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====== Custom CSS ======
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00CC96, #636EFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)


# ====== Cached Data Loading ======
@st.cache_data
def load_and_process():
    loader = WaterQualityDataLoader("data/water_quality.csv")
    df = loader.load()
    df = WQICalculator().calculate(df)
    detector = AnomalyDetector(contamination=0.05)
    df = detector.fit_predict(df)
    return df


# ====== Header ======
st.markdown('<h1 class="main-header">🌊 AquaInsight</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">AI-Powered Freshwater Quality Monitoring for One Health</p>',
    unsafe_allow_html=True,
)

# ====== Load Data ======
with st.spinner("🔄 Loading water quality data..."):
    df = load_and_process()

# ====== Sidebar Filters ======
st.sidebar.header("🔍 Filters")

stations = ["All Stations"] + sorted(df["station_id"].unique().tolist())
selected_station = st.sidebar.selectbox("Station", stations)

min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Apply filters
filtered = df.copy()
if selected_station != "All Stations":
    filtered = filtered[filtered["station_id"] == selected_station]

if len(date_range) == 2:
    filtered = filtered[
        (filtered["date"].dt.date >= date_range[0])
        & (filtered["date"].dt.date <= date_range[1])
    ]

# ====== KPI Metrics ======
st.markdown("### 📊 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Records", f"{len(filtered):,}")
with col2:
    st.metric("Mean WQI", f"{filtered['wqi'].mean():.2f}", 
              delta=f"{filtered['wqi'].mean() - df['wqi'].mean():.2f}")
with col3:
    st.metric("Stations", filtered["station_id"].nunique())
with col4:
    st.metric("⚠️ Anomalies", int(filtered["is_anomaly"].sum()))
with col5:
    alerts_count = len(filtered[filtered["wqi"] < 50])
    st.metric("🚨 Alerts", alerts_count, delta_color="inverse")

st.markdown("---")

# ====== Tabs ======
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Time Series",
    "🗺️ Distribution",
    "⚠️ Anomalies",
    "🔬 Parameters",
    "📋 Data",
])

# ====== Tab 1: Time Series ======
with tab1:
    st.markdown("### Water Quality Index Over Time")
    fig = px.line(
        filtered,
        x="date",
        y="wqi",
        color="station_id",
        labels={"wqi": "WQI", "date": "Date", "station_id": "Station"},
    )
    fig.add_hline(y=70, line_dash="dash", line_color="green",
                  annotation_text="Good (70)")
    fig.add_hline(y=50, line_dash="dash", line_color="orange",
                  annotation_text="Fair (50)")
    fig.update_layout(template="plotly_white", height=500, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ====== Tab 2: Distribution ======
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### WQI Categories")
        counts = filtered["category"].value_counts().reset_index()
        counts.columns = ["Category", "Count"]
        fig = px.pie(
            counts,
            names="Category",
            values="Count",
            hole=0.4,
            color="Category",
            color_discrete_map={
                "Excellent": "#00CC96",
                "Good": "#19D3F3",
                "Fair": "#FFA15A",
                "Poor": "#EF553B",
                "Very Poor": "#636EFA",
            },
        )
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Average WQI per Station")
        means = filtered.groupby("station_id")["wqi"].mean().reset_index()
        means.columns = ["Station", "Mean WQI"]
        means = means.sort_values("Mean WQI")
        fig = px.bar(
            means,
            x="Mean WQI",
            y="Station",
            orientation="h",
            color="Mean WQI",
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

# ====== Tab 3: Anomalies ======
with tab3:
    st.markdown("### Anomaly Detection (Isolation Forest)")
    anomalies = filtered[filtered["is_anomaly"]]
    st.info(f"Detected **{len(anomalies)}** anomalies out of **{len(filtered)}** records")

    fig = px.scatter(
        filtered,
        x="date",
        y="wqi",
        color="anomaly_label",
        symbol="station_id",
        color_discrete_map={
            "⚠️ Anomaly": "#EF553B",
            "✅ Normal": "#00CC96",
        },
        labels={"anomaly_label": "Status"},
    )
    fig.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

    if len(anomalies) > 0:
        st.markdown("### 🚨 Anomaly Records")
        st.dataframe(
            anomalies[["date", "station_id", "wqi", "ph",
                       "dissolved_oxygen", "turbidity", "anomaly_score"]]
            .sort_values("anomaly_score")
            .head(20),
            use_container_width=True,
        )

# ====== Tab 4: Parameters ======
with tab4:
    param = st.selectbox(
        "Select Parameter",
        ["ph", "dissolved_oxygen", "turbidity", "temperature", "nitrate", "phosphate"],
    )
    fig = px.box(
        filtered,
        x="station_id",
        y=param,
        color="station_id",
        labels={"station_id": "Station"},
    )
    fig.update_layout(template="plotly_white", height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ====== Tab 5: Raw Data ======
with tab5:
    st.markdown("### 📋 Processed Data")
    st.dataframe(filtered.head(100), use_container_width=True)
    st.download_button(
        "📥 Download CSV",
        filtered.to_csv(index=False),
        "aquainsight_data.csv",
        "text/csv",
    )

# ====== Footer ======
st.markdown("---")
st.markdown(
    "<center><small>🌊 <b>AquaInsight</b> · OneAquaHealth Hackathon 2025 · Built with ❤️ and Streamlit</small></center>",
    unsafe_allow_html=True,
)
