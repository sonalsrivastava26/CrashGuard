from pathlib import Path
from safest_route_utils import build_and_route

# Example: MG Road Metro -> Koramangala 80ft
origin_lat, origin_lon = 12.9758, 77.6046
dest_lat, dest_lon     = 12.9344, 77.6271

alpha = 1.0   # distance weight
beta  = 300.0 # risk weight (↑ to avoid hotspots more)

out_path, clusters, n = build_and_route(
    origin_lat=origin_lat, origin_lon=origin_lon,
    dest_lat=dest_lat, dest_lon=dest_lon,
    place="Bengaluru, Karnataka, India",
    alpha=alpha, beta=beta,
    data_path=Path("../outputs/data_prepared.csv"),
    hotspots_csv=Path("../outputs/hotspots.csv"),
    save_map_path=Path("../outputs/route_map.html"),
    recompute_hotspots=False  # set True to recompute DBSCAN inside
)

print(f"[OK] Accidents used: {n:,}")
print(f"[OK] Hotspots: {len(clusters)}")
print(f"[OK] Map saved: {Path(out_path).absolute()}")
