import os
from pathlib import Path
import streamlit as st

from safest_route_utils import build_and_route

st.set_page_config(page_title="Bangalore Safest Route", layout="wide")
st.title("🚦 Bangalore Safest Route (OpenStreetMap + DBSCAN)")

st.markdown(
    "This app uses **OpenStreetMap**, **DBSCAN** for hotspots, "
    "and **risk-aware routing** to find a safer path."
)

with st.sidebar:
    st.header("Route Inputs")
    origin_lat = st.number_input("Origin Latitude", value=12.9758, format="%.6f")
    origin_lon = st.number_input("Origin Longitude", value=77.6046, format="%.6f")
    dest_lat = st.number_input("Destination Latitude", value=12.9344, format="%.6f")
    dest_lon = st.number_input("Destination Longitude", value=77.6271, format="%.6f")

    st.header("Tradeoff")
    alpha = st.slider("Distance weight (alpha)", 0.1, 5.0, 1.0, 0.1)
    beta = st.slider("Risk weight (beta)", 0.0, 1000.0, 300.0, 50.0)

    st.header("Hotspots (DBSCAN)")
    recompute = st.checkbox("Recompute hotspots now", value=False)
    eps_m = st.slider("DBSCAN eps (meters)", 50, 300, 120, 10)
    min_samples = st.slider("DBSCAN min_samples", 5, 100, 25, 5)

    run_btn = st.button("Compute Safest Route")

st.markdown("---")

if run_btn:
    with st.spinner("Computing route and rendering map..."):
        out_path, clusters, n = build_and_route(
            origin_lat, origin_lon, dest_lat, dest_lon,
            place="Bengaluru, Karnataka, India",
            alpha=alpha, beta=beta,
            data_path=Path("../outputs/data_prepared.csv"),
            hotspots_csv=Path("../outputs/hotspots.csv"),
            save_map_path=Path("../outputs/route_map.html"),
            recompute_hotspots=recompute,
            eps_m=eps_m, min_samples=min_samples
        )
    st.success(f"Done! Accidents used: {n:,} • Hotspots: {len(clusters)}")
    st.caption(f"Map file: {out_path}")

    if Path(out_path).exists():
        html = Path(out_path).read_text(encoding="utf-8")
        st.components.v1.html(html, height=680, scrolling=True)
else:
    st.info("Fill inputs on the left and click **Compute Safest Route**.")

st.markdown("---")
st.write("**Tip:** Increase **beta** to avoid hotspots more aggressively. Tune DBSCAN for coarser/finer hotspot detection.")

