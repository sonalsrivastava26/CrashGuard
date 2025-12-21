import sys
from pathlib import Path
from safest_route_utils import build_and_route

def main():
    # Coordinates for MG Road Metro to Koramangala 80ft, Bangalore
    origin_lat, origin_lon = 12.9758, 77.6046
    dest_lat, dest_lon = 12.9344, 77.6271
    alpha = 1.0      # Distance weight
    beta = 300.0     # Risk weight (higher to avoid hotspots more)

    data_path = Path("../outputs/data_prepared.csv")
    hotspots_csv = Path("../outputs/hotspots_map.csv")
    save_map_path = Path("../outputs/route_map.html")

    try:
        out_path, clusters, n = build_and_route(
            origin_lat=origin_lat, origin_lon=origin_lon,
            dest_lat=dest_lat, dest_lon=dest_lon,
            place="Bengaluru, Karnataka, India",
            alpha=alpha, beta=beta,
            data_path=data_path,
            hotspots_csv=hotspots_csv,
            save_map_path=save_map_path,
            recompute_hotspots=False
        )
        print(f"[OK] Accidents used: {n:,}")
        print(f"[OK] Hotspots: {len(clusters)}")
        print(f"[OK] Map saved: {Path(out_path).absolute()}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
