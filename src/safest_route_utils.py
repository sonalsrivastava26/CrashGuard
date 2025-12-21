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
import heapq
from shapely.geometry import LineString


R_EARTH = 6371000.0  # meters


# Central config defaults
DEFAULT_EPS_M = 200.0
DEFAULT_MIN_SAMPLES = 15
DEFAULT_PLACE = "Bengaluru, Karnataka, India"
DEFAULT_HOTSPOTS_FILE = Path("../outputs/hotspots_map.csv")


def load_accident_data(path: str | Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    except FileNotFoundError:
        raise FileNotFoundError(f"Accident data file '{path}' not found.")
    df = df.dropna(subset=["lat", "lon"]).copy()
    return df


def detect_hotspots_dbscan(df: pd.DataFrame, eps_m: float = DEFAULT_EPS_M, min_samples: int = DEFAULT_MIN_SAMPLES) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    coords_rad = np.radians(df[["lat", "lon"]].values)
    db = DBSCAN(eps=eps_m / R_EARTH, min_samples=min_samples, metric="haversine", n_jobs=-1)
    labels = db.fit_predict(coords_rad)
    df = df.copy()
    df["cluster_id"] = labels
    clusters = (
        df[df["cluster_id"] >= 0]
        .groupby("cluster_id")
        .agg(lat=("lat", "mean"), lon=("lon", "mean"),
             count=("cluster_id", "size"), avg_speed=("speed_kmh", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return clusters


def build_graph(place: str = DEFAULT_PLACE) -> nx.MultiDiGraph:
    G = ox.graph_from_place(place, network_type="drive")
    G = add_edge_lengths(G)
    return G


def _hotspot_tree(clusters: pd.DataFrame):
    if clusters is None or len(clusters) == 0:
        return None, None, None
    pts_deg = clusters[["lat", "lon"]].values
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
    lat_u = G.nodes[u].get("y", None)
    lat_v = G.nodes[v].get("y", None)
    lon_u = G.nodes[u].get("x", None)
    lon_v = G.nodes[v].get("x", None)
    if None in (lat_u, lat_v, lon_u, lon_v):
        raise ValueError(f"Missing node coordinates for edge ({u}, {v})")
    lat = (lat_u + lat_v) / 2
    lon = (lon_u + lon_v) / 2
    return float(lat), float(lon)


def attach_risk_to_graph(G: nx.MultiDiGraph, clusters: pd.DataFrame,
                         lam_m: float = 150.0, k_neigh: int = 5,
                         origin_coords: tuple[float,float] = None,
                         dest_coords: tuple[float,float] = None,
                         origin_weather_factor: float = 1.0,
                         dest_weather_factor: float = 1.0,
                         weather_risk_radius_m: float = 2000.0) -> nx.MultiDiGraph:
    tree, pts_rad, weights = _hotspot_tree(clusters)
    if tree is None:
        for _, _, _, d in G.edges(keys=True, data=True):
            d["risk"] = 0.0
        return G

    k = int(min(k_neigh, len(clusters)))

    import numpy as np
    origin_rad = np.radians(origin_coords) if origin_coords else None
    dest_rad = np.radians(dest_coords) if dest_coords else None

    for u, v, kkey, data in G.edges(keys=True, data=True):
        try:
            lat, lon = _edge_midpoint_latlon(G, u, v, data)
        except Exception:
            data["risk"] = 0.0
            continue

        dists, idxs = tree.query(np.radians([[lat, lon]]), k=k)
        dists_m = dists[0] * R_EARTH
        w = weights[idxs[0]]
        base_risk = float(np.sum(w * np.exp(-dists_m / lam_m)))

        edge_rad = np.radians([lat, lon])
        weather_mult = 1.0

        if origin_rad is not None:
            dist_origin = np.linalg.norm(edge_rad - origin_rad) * R_EARTH
            if dist_origin <= weather_risk_radius_m:
                weather_mult = max(weather_mult, origin_weather_factor)

        if dest_rad is not None:
            dist_dest = np.linalg.norm(edge_rad - dest_rad) * R_EARTH
            if dist_dest <= weather_risk_radius_m:
                weather_mult = max(weather_mult, dest_weather_factor)

        data["risk"] = base_risk * weather_mult

    return G


def bmssp(G, sources, bound_B, weight="safety_weight"):
    dist = {v: float('inf') for v in G.nodes}
    pq = []
    for s in sources:
        dist[s] = 0
        heapq.heappush(pq, (0, s))
    complete = set()
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if d >= bound_B:
            break
        complete.add(u)
        for v in G.successors(u):
            for key in G[u][v]:
                w = G[u][v][key].get(weight, 1.0)
                nd = d + w
                if nd < dist[v] and nd < bound_B:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
    return {v: dist[v] for v in complete}


def safest_path_bmssp(G: nx.MultiDiGraph, origin_latlon: tuple[float, float],
                      dest_latlon: tuple[float, float], alpha: float = 1.0, beta: float = 20.0,
                      bound_B: float = 10000):
    orig = ox.distance.nearest_nodes(G, X=[origin_latlon[1]], Y=[origin_latlon[0]])[0]
    dest = ox.distance.nearest_nodes(G, X=[dest_latlon[1]], Y=[dest_latlon[0]])[0]
    lengths = [d.get("length", 1.0) for _, _, _, d in G.edges(keys=True, data=True)]
    risks = [d.get("risk", 0.0) for _, _, _, d in G.edges(keys=True, data=True)]
    max_len = max(lengths) if lengths else 1.0
    max_risk = max(risks) if risks else 1.0
    for _, _, _, d in G.edges(keys=True, data=True):
        norm_len = d.get("length", 1.0) / max_len
        norm_risk = d.get("risk", 0.0) / (max_risk + 1e-9)
        d["safety_weight"] = alpha * norm_len + beta * norm_risk
        if norm_risk > 0.7:
            d["safety_weight"] += 50
    dist_map = bmssp(G, [orig], bound_B, weight="safety_weight")
    if dest not in dist_map:
        raise Exception("Destination not reachable within bound_B")
    path = nx.shortest_path(G, source=orig, target=dest, weight="safety_weight")
    return path


def folium_map_with_route(G, path_nodes, clusters, accidents_df=None, save_path=None, route_color="green"):
    if path_nodes:
        start_lat, start_lon = G.nodes[path_nodes[0]]['y'], G.nodes[path_nodes[0]]['x']
    else:
        start_lat, start_lon = 12.9716, 77.5946

    m = folium.Map(location=[start_lat, start_lon], zoom_start=13, tiles="OpenStreetMap")

    blue_gradient = {
        0.0: 'navy',
        0.3: 'blue',
        0.6: 'deepskyblue',
        1.0: 'aqua'
    }

    if accidents_df is not None and len(accidents_df) > 0:
        pts = accidents_df[["lat","lon"]].dropna().values.tolist()
        HeatMap(
            pts,
            radius=15,
            blur=18,
            max_zoom=15,
            gradient=blue_gradient,
            min_opacity=0.3
        ).add_to(m)

    mc = MarkerCluster(name='Accident Hotspots').add_to(m)
    for _, r in clusters.iterrows():
        folium.Marker(
            location=[r['lat'], r['lon']],
            popup=folium.Popup(f"Accident count: {int(r['count'])}", max_width=200),
            tooltip=f"{int(r['count'])} accidents"
        ).add_to(mc)

    if path_nodes and len(path_nodes) > 1:
        coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path_nodes]
        folium.PolyLine(coords, color=route_color, weight=6, opacity=0.9, tooltip="Safest route").add_to(m)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(save_path)

    return m


def build_and_route(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float,
                    place: str = DEFAULT_PLACE,
                    alpha: float = 1.0, beta: float = 20.0,
                    data_path: str | Path = Path("../outputs/data_prepared.csv"),
                    hotspots_csv: str | Path = DEFAULT_HOTSPOTS_FILE,
                    save_map_path: str | Path = Path("../outputs/route_map.html"),
                    recompute_hotspots: bool = False,
                    eps_m: float = DEFAULT_EPS_M, min_samples: int = DEFAULT_MIN_SAMPLES,
                    bound_B: float = 10000):
    try:
        df = load_accident_data(data_path)
        if recompute_hotspots or not Path(hotspots_csv).exists():
            clusters = detect_hotspots_dbscan(df, eps_m=eps_m, min_samples=min_samples)
            clusters.to_csv(hotspots_csv, index=False)
        else:
            clusters = pd.read_csv(hotspots_csv)
        G = build_graph(place=place)
        G = attach_risk_to_graph(G, clusters)
        path = safest_path_bmssp(G, (origin_lat, origin_lon), (dest_lat, dest_lon),
                                alpha=alpha, beta=beta, bound_B=bound_B)
        folium_map_with_route(G, path, clusters, accidents_df=df, save_path=save_map_path)
        return save_map_path, clusters, len(df)
    except Exception as ex:
        raise RuntimeError(f"Error in build_and_route pipeline: {ex}")
