import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import HeatMap, MarkerCluster
from pathlib import Path

R_EARTH = 6371000.0
DATA = Path("../outputs/data_prepared.csv")
OUT_DIR = Path("../outputs")
HOTSPOTS = OUT_DIR / "hotspots.csv"
MAP = OUT_DIR / "hotspots_map.html"

# --- Tunables ---
EPS_M = 120          # DBSCAN radius in meters
MIN_SAMPLES = 25     # min points per cluster
HEAT_SAMPLE = 100000 # cap points in heatmap for performance

df = pd.read_csv(DATA, parse_dates=["timestamp_utc"])
coords_rad = np.radians(df[["lat","lon"]].values)

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
    .agg(lat=("lat","mean"), lon=("lon","mean"),
         count=("cluster_id","size"), avg_speed=("speed_kmh","mean"))
    .reset_index()
    .sort_values("count", ascending=False)
)
clusters.to_csv(HOTSPOTS, index=False)
print(f"[OK] Hotspots: {len(clusters)} -> {HOTSPOTS}")

# Map with OSM tiles
center = [df["lat"].median(), df["lon"].median()]
m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

# Heatmap
pts = df[["lat","lon"]]
if len(pts) > HEAT_SAMPLE:
    pts = pts.sample(HEAT_SAMPLE, random_state=42)
HeatMap(pts.values.tolist(), radius=10, blur=12, max_zoom=17).add_to(m)

# Hotspot markers
mc = MarkerCluster().add_to(m)
for _, r in clusters.iterrows():
    folium.CircleMarker(
        location=[r.lat, r.lon],
        radius=6 + min(14, r["count"]/150),
        popup=folium.Popup(
            f"Hotspot #{int(r['cluster_id'])}<br>"
            f"Events: {int(r['count'])}<br>"
            f"Avg speed: {r['avg_speed']:.1f} km/h",
            max_width=260
        ),
        tooltip=f"Hotspot {int(r['cluster_id'])} • {int(r['count'])} events",
        fill=True, fill_opacity=0.75
    ).add_to(mc)

m.save(MAP)
print(f"[OK] Map -> {MAP.absolute()}")
