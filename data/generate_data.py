"""Generate synthetic water quality data for AquaInsight."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# Generate 365 days of data for 5 stations
stations = ["Station_A", "Station_B", "Station_C", "Station_D", "Station_E"]
dates = pd.date_range(start="2024-01-01", periods=365, freq="D")

records = []
for station in stations:
    base_ph = np.random.uniform(6.5, 7.5)
    base_do = np.random.uniform(6.0, 9.0)
    base_turbidity = np.random.uniform(2.0, 8.0)
    base_temp = np.random.uniform(15.0, 25.0)
    base_nitrate = np.random.uniform(1.0, 5.0)
    base_phosphate = np.random.uniform(0.1, 1.0)
    
    for date in dates:
        # Add seasonality + noise
        day_of_year = date.dayofyear
        seasonal = 2 * np.sin(2 * np.pi * day_of_year / 365)
        
        ph = np.clip(base_ph + 0.3 * np.random.randn(), 5.0, 9.5)
        do = np.clip(base_do + seasonal * 0.5 + 0.5 * np.random.randn(), 2.0, 12.0)
        turbidity = np.clip(base_turbidity + 1.5 * np.random.randn(), 0.5, 20.0)
        temp = base_temp + seasonal * 3 + 1.0 * np.random.randn()
        nitrate = np.clip(base_nitrate + 0.5 * np.random.randn(), 0.0, 15.0)
        phosphate = np.clip(base_phosphate + 0.2 * np.random.randn(), 0.0, 3.0)
        
        # Inject anomalies (5% chance)
        if np.random.random() < 0.05:
            ph = np.clip(ph + np.random.choice([-1.5, 1.5]), 3.0, 11.0)
            do = np.clip(do + np.random.choice([-3.0, 3.0]), 0.5, 15.0)
            turbidity = np.clip(turbidity + 10, 0, 50)
        
        records.append({
            "station_id": station,
            "date": date,
            "ph": round(ph, 2),
            "dissolved_oxygen": round(do, 2),
            "turbidity": round(turbidity, 2),
            "temperature": round(temp, 2),
            "nitrate": round(nitrate, 2),
            "phosphate": round(phosphate, 2),
        })

df = pd.DataFrame(records)
df.to_csv("data/water_quality.csv", index=False)

print(f"✅ Generated {len(df)} records")
print(f"📊 Stations: {len(stations)}")
print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
print(f"📁 Saved: data/water_quality.csv")
