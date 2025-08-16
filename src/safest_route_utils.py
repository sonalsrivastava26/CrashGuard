"""
Safest route utilities (safety prioritized):
- Loads accidents, detects DBSCAN hotspots (or reads precomputed).
- Builds a drivable graph from OpenStreetMap (free).
- Attaches a risk score to each edge based on distance to hotspots.
- Computes a route minimizing alpha*length + beta*risk (with strong bias to safety).
- Saves an interactive Folium map with route + hotspots.

No Google APIs used.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import folium
from folium.plugins import HeatMap, MarkerCluster
from osmnx.distance import add_edge_lengths
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
import osmnx as ox

R_EARTH = 6371000.0  # meters

# ----------------------------
# Data loading / hotspots
# ----------------------------
def load_accident_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    df = df.dropna(subset=["lat", "lon"]).copy()
    return df

def detect_hotspots_dbscan(df: pd.DataFrame, eps_m: float = 120, min_samples: int = 25) -> pd.DataFrame:
    coords_rad = np.radians(df[["lat","lon"]].values)
    db = DBSCAN(eps=eps_m/R_EARTH, min_samples=min_samples, metric="haversine", n_jobs=-1)
    labels = db.fit_predict(coords_rad)
    df = df.copy()
    df["cluster_id"] = labels
    clusters = (
        df[df["cluster_id"] >= 0]
        .groupby("cluster_id")
        .agg(lat=("lat","mean"), lon=("lon","mean"),
             count=("cluster_id","size"), avg_speed=("speed_kmh","mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return clusters

# ----------------------------
# Graph + risk scoring
# ----------------------------
def build_graph(place: str = "Bengaluru, Karnataka, India") -> nx.MultiDiGraph:
    G = ox.graph_from_place(place, network_type="drive")
    G = add_edge_lengths(G)
    return G

def _hotspot_tree(clusters: pd.DataFrame):
    if clusters is None or len(clusters) == 0:
        return None, None, None
    pts_deg = clusters[["lat","lon"]].values
    pts_rad = np.radians(pts_deg)
    tree = BallTree(pts_rad, metric="haversine")
    weights = clusters["count"].to_numpy(dtype=float)
    if len(weights) > 0:
        wmin, wmax = weights.min(), weights.max()
        weights = (weights - wmin) / (wmax - wmin + 1e-9)
    return tree, pts_rad, weights

def _edge_midpoint_latlon(G: nx.MultiDiGraph, u, v, data) -> tuple[float, float]:
    if "geometry" in data and data["geometry"] is not None:
        geom = data["geometry"]
        return float(geom.centroid.y), float(geom.centroid.x)
    lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
    lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
    return float(lat), float(lon)

def attach_risk_to_graph(G: nx.MultiDiGraph, clusters: pd.DataFrame,
                         lam_m: float = 150.0, k_neigh: int = 5) -> nx.MultiDiGraph:
    tree, _, weights = _hotspot_tree(clusters)
    if tree is None:
        for _, _, _, d in G.edges(keys=True, data=True):
            d["risk"] = 0.0
        return G

    k = int(min(k_neigh, len(clusters)))
    for u, v, kkey, data in G.edges(keys=True, data=True):
        lat, lon = _edge_midpoint_latlon(G, u, v, data)
        dists, idxs = tree.query(np.radians([[lat, lon]]), k=k)
        dists_m = dists[0] * R_EARTH
        w = weights[idxs[0]]
        risk = float(np.sum(w * np.exp(-dists_m / lam_m)))
        data["risk"] = risk
    return G

# ----------------------------
# Routing + map rendering
# ----------------------------
def safest_path(G: nx.MultiDiGraph, origin_latlon: tuple[float, float],
                dest_latlon: tuple[float, float], alpha: float = 1.0, beta: float = 20.0):
    """
    Strongly prioritize safety over distance.
    Minimize alpha*normalized_length + beta*normalized_risk (+ extra hotspot penalty).
    """
    orig = ox.distance.nearest_nodes(G, X=[origin_latlon[1]], Y=[origin_latlon[0]])[0]
    dest = ox.distance.nearest_nodes(G, X=[dest_latlon[1]], Y=[dest_latlon[0]])[0]

    lengths = [d.get("length", 1.0) for _,_,_,d in G.edges(keys=True, data=True)]
    risks   = [d.get("risk", 0.0) for _,_,_,d in G.edges(keys=True, data=True)]
    max_len = max(lengths) if lengths else 1.0
    max_risk = max(risks) if risks else 1.0

    for _, _, _, d in G.edges(keys=True, data=True):
        norm_len = d.get("length", 1.0) / max_len
        norm_risk = d.get("risk", 0.0) / (max_risk + 1e-9)

        d["safety_weight"] = alpha * norm_len + beta * norm_risk

        # Big penalty if edge passes through a dense hotspot
        if norm_risk > 0.7:
            d["safety_weight"] += 50

    path = nx.shortest_path(G, source=orig, target=dest, weight="safety_weight")
    return path

def folium_map_with_route(G: nx.MultiDiGraph, path_nodes: list[int],
                          clusters: pd.DataFrame, accidents_df: pd.DataFrame | None,
                          save_path: str | Path) -> None:
    if path_nodes:
        lats = [G.nodes[n]["y"] for n in path_nodes]
        lons = [G.nodes[n]["x"] for n in path_nodes]
        center = [float(np.mean(lats)), float(np.mean(lons))]
    else:
        center = [12.9716, 77.5946]

    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    if accidents_df is not None and len(accidents_df) > 0:
        pts = accidents_df[["lat","lon"]].dropna().values.tolist()
        HeatMap(pts, radius=7, blur=11, max_zoom=17).add_to(m)

    if clusters is not None and len(clusters) > 0:
        mc = MarkerCluster().add_to(m)
        for _, r in clusters.iterrows():
            folium.CircleMarker(
                location=[r.lat, r.lon],
                radius=6 + min(14, r["count"]/150),
                popup=folium.Popup(
                    f"Hotspot #{int(r['cluster_id']) if 'cluster_id' in r else ''}"
                    f"<br>Events: {int(r['count'])}"
                    + (f"<br>Avg speed: {r['avg_speed']:.1f} km/h" if 'avg_speed' in r else ""),
                    max_width=260
                ),
                tooltip=f"Hotspot • {int(r['count'])} events",
                fill=True, fill_opacity=0.75
            ).add_to(mc)

    if path_nodes and len(path_nodes) > 1:
        coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path_nodes]
        folium.PolyLine(coords, weight=6, opacity=0.9, color="green", tooltip="Safest route").add_to(m)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    m.save(str(save_path))

# ----------------------------
# End-to-end convenience
# ----------------------------
def build_and_route(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float,
                    place: str = "Bengaluru, Karnataka, India",
                    alpha: float = 1.0, beta: float = 20.0,
                    data_path: str | Path = "outputs/data_prepared.csv",
                    hotspots_csv: str | Path = "outputs/hotspots.csv",
                    save_map_path: str | Path = "outputs/route_map.html",
                    recompute_hotspots: bool = False,
                    eps_m: float = 120, min_samples: int = 25):

    df = load_accident_data(data_path)

    if recompute_hotspots or not Path(hotspots_csv).exists():
        clusters = detect_hotspots_dbscan(df, eps_m=eps_m, min_samples=min_samples)
        clusters.to_csv(hotspots_csv, index=False)
    else:
        clusters = pd.read_csv(hotspots_csv)

    G = build_graph(place=place)
    G = attach_risk_to_graph(G, clusters)

    path = safest_path(G, (origin_lat, origin_lon), (dest_lat, dest_lon), alpha=alpha, beta=beta)
    folium_map_with_route(G, path, clusters, accidents_df=df, save_path=save_map_path)

    return save_map_path, clusters, len(df)
