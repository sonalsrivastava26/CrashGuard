import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import HeatMap, MarkerCluster
from pathlib import Path

"""
Clusters accident data with DBSCAN to detect hotspots and creates an interactive map.
"""

# Configuration
R_EARTH = 6371000.0           # Earth radius in meters
DATA = Path("../outputs/data_prepared.csv")
OUT_DIR = Path("../outputs")
HOTSPOTS_CSV = OUT_DIR / "hotspots_map.csv"   # Unified name for hotspot CSV
MAP_HTML = OUT_DIR / "hotspots_map.html"

EPS_M = 200                   # DBSCAN radius in meters (matches safest_route_utils default)
MIN_SAMPLES = 15
HEAT_SAMPLE = 100000          # Max points to render on heatmap for performance

def main():
    try:
        df = pd.read_csv(DATA, parse_dates=["timestamp_utc"])
    except FileNotFoundError:
        raise FileNotFoundError(f"Input data file '{DATA}' not found.")

    if not {"lat", "lon", "speed_kmh"}.issubset(df.columns):
        raise ValueError("Input data missing required columns: 'lat', 'lon', or 'speed_kmh'.")

    coords_rad = np.radians(df[["lat", "lon"]].values)

    db = DBSCAN(
        eps=EPS_M / R_EARTH,
        min_samples=MIN_SAMPLES,
        metric="haversine",
        n_jobs=-1
    )
    labels = db.fit_predict(coords_rad)
    df["cluster_id"] = labels

    clusters = (
        df[df["cluster_id"] >= 0]
        .groupby("cluster_id")
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            count=("cluster_id", "size"),
            avg_speed=("speed_kmh", "mean")
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    clusters.to_csv(HOTSPOTS_CSV, index=False)
    print(f"[OK] Hotspots: {len(clusters)} -> {HOTSPOTS_CSV}")

    # Map creation
    center = [df["lat"].median(), df["lon"].median()]
    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

    # Heatmap (sampled for large data)
    pts = df[["lat", "lon"]]
    if len(pts) > HEAT_SAMPLE:
        pts = pts.sample(HEAT_SAMPLE, random_state=42)
    HeatMap(pts.values.tolist(), radius=10, blur=12, max_zoom=17).add_to(m)

    # Hotspot markers with popup info
    mc = MarkerCluster().add_to(m)
    for _, r in clusters.iterrows():
        folium.CircleMarker(
            location=[r.lat, r.lon],
            radius=6 + min(14, r["count"] / 150),
            popup=folium.Popup(
                f"Hotspot #{int(r['cluster_id'])}<br>"
                f"Events: {int(r['count'])}<br>"
                f"Avg speed: {r['avg_speed']:.1f} km/h",
                max_width=260
            ),
            tooltip=f"Hotspot {int(r['cluster_id'])} • {int(r['count'])} events",
            fill=True, fill_opacity=0.75
        ).add_to(mc)

    m.save(MAP_HTML)
    print(f"[OK] Map saved at {MAP_HTML.absolute()}")

if __name__ == "__main__":
    main()
