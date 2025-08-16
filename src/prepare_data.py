import pandas as pd
from pathlib import Path

DATA_IN = Path("../data/bangalore-cas-alerts.csv")
OUT_DIR = Path("../outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUT = OUT_DIR / "data_prepared.csv"

df = pd.read_csv(DATA_IN)

df = df.rename(columns={
    "deviceCode_location_latitude": "lat",
    "deviceCode_location_longitude": "lon",
    "deviceCode_location_wardName": "ward",
    "deviceCode_pyld_alarmType": "alarm_type",
    "deviceCode_pyld_speed": "speed_kmh",
    "deviceCode_time_recordedTime_$date": "timestamp_utc",
})

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
df["year"] = df["timestamp_utc"].dt.year
df["month"] = df["timestamp_utc"].dt.month
df["hour"] = df["timestamp_utc"].dt.hour
df["dow"] = df["timestamp_utc"].dt.dayofweek

# Keep roughly-Bengaluru bounds
df = df[(df["lat"].between(12.7, 13.2)) & (df["lon"].between(77.3, 77.9))].copy()

cols = ["lat","lon","ward","alarm_type","speed_kmh","timestamp_utc","year","month","hour","dow"]
df[cols].to_csv(DATA_OUT, index=False)
print(f"[OK] Wrote {DATA_OUT} with {len(df):,} rows")