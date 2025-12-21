import pandas as pd
from pathlib import Path

"""
Prepares raw Bangalore accident alerts data for hotspot and routing analysis.
- Renames columns
- Parses timestamps
- Filters to Bangalore bounding box
- Saves cleaned data CSV for downstream usage
"""

DATA_IN = Path("../data/bangalore-cas-alerts.csv")
OUT_DIR = Path("../outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUT = OUT_DIR / "data_prepared.csv"

REQUIRED_COLUMNS = [
    "deviceCode_location_latitude",
    "deviceCode_location_longitude",
    "deviceCode_pyld_alarmType",
    "deviceCode_pyld_speed",
    "deviceCode_time_recordedTime_$date"
]

def main():
    try:
        df = pd.read_csv(DATA_IN)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input data file '{DATA_IN}' not found.")

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in input data: {missing_cols}")

    df = df.rename(columns={
        "deviceCode_location_latitude": "lat",
        "deviceCode_location_longitude": "lon",
        "deviceCode_location_wardName": "ward",
        "deviceCode_pyld_alarmType": "alarm_type",
        "deviceCode_pyld_speed": "speed_kmh",
        "deviceCode_time_recordedTime
