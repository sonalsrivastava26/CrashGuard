import time
from pathlib import Path
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import pandas as pd
from weather_api import get_current_weather

# Import safest_route_utils functions
from safest_route_utils import (
    build_and_route, load_accident_data, detect_hotspots_dbscan,
    build_graph, attach_risk_to_graph, safest_path_bmssp, folium_map_with_route
)

DATA_PATH = Path("../outputs/data_prepared.csv")
HOTSPOTS_PATH = Path("../outputs/hotspots_map.csv")
MAP_PATH = Path("../outputs/route_map.html")

st.set_page_config(page_title="Bangalore Safest Route", layout="wide")
st.title("🚦 Bangalore Safest Route (OpenStreetMap + DBSCAN)")
st.markdown(
    "This app uses **OpenStreetMap**, **DBSCAN** for hotspots, "
    "and **risk-aware routing with live weather risk adjustment** to find a safer path."
)

geolocator = Nominatim(user_agent="bangalore_safest_route_app")

@st.cache_data(show_spinner=False)
def geocode_location_cached(place_name: str):
    try:
        location = geolocator.geocode(place_name + ", Bengaluru, Karnataka, India", timeout=10)
        if location:
            return location.latitude, location.longitude
        return None, None
    except (GeocoderTimedOut, GeocoderServiceError):
        return None, None

@st.cache_data(show_spinner=False)
def load_accident_data_cached(path):
    return load_accident_data(path)

@st.cache_data(show_spinner=False)
def detect_hotspots_cached(df, eps_m, min_samples):
    start = time.perf_counter()
    clusters = detect_hotspots_dbscan(df, eps_m=eps_m, min_samples=min_samples)
    elapsed = time.perf_counter() - start
    return clusters, elapsed

@st.cache_resource(show_spinner=False)
def build_graph_cached(place):
    start = time.perf_counter()
    G = build_graph(place)
    elapsed = time.perf_counter() - start
    return G, elapsed

@st.cache_resource(show_spinner=False)
def attach_risk_cached(_G, clusters,
                       origin_coords=None,
                       dest_coords=None,
                       origin_weather_factor=1.0,
                       dest_weather_factor=1.0):
    from safest_route_utils import attach_risk_to_graph  # delayed import for clarity
    start = time.perf_counter()
    G = attach_risk_to_graph(_G, clusters,
                             origin_coords=origin_coords,
                             dest_coords=dest_coords,
                             origin_weather_factor=origin_weather_factor,
                             dest_weather_factor=dest_weather_factor)
    elapsed = time.perf_counter() - start
    return G, elapsed

with st.sidebar:
    st.header("Route Inputs")
    origin_area = st.text_input("Origin area name", value="Hoodi")
    dest_area = st.text_input("Destination area name", value="Koramangala")

    st.header("Route Preference")
    route_profile = st.radio("Select your route preference", 
                             ["Balanced", "Shortest Route (High Risk)", "Safest Route (Longer)", "Custom"])

    if route_profile == "Balanced":
        alpha_val, beta_val = 1.0, 20.0
        st.markdown("Balanced tradeoff between route length and safety.")
    elif route_profile == "Shortest Route (High Risk)":
        alpha_val, beta_val = 5.0, 0.0
        st.markdown("Prioritize shortest distance, ignoring hotspot risks.")
    elif route_profile == "Safest Route (Longer)":
        alpha_val, beta_val = 0.1, 500.0
        st.markdown("Strongly avoid risky areas, accept longer travel time.")
    else:  # Custom profile
        alpha_val = st.slider("Distance weight (alpha)", 0.1, 5.0, 1.0, 0.1,
                              help="How much to prioritize shorter routes.")
        beta_val = st.slider("Risk weight (beta)", 0.0, 1000.0, 20.0, 50.0,
                             help="How strongly to avoid hotspots.")

    st.header("Hotspots (DBSCAN)")
    recompute = st.checkbox("Recompute hotspots now", value=False)
    eps_m = st.slider("DBSCAN eps (meters)", 50, 300, 200, 10,
                      help="Radius for accident cluster detection (larger = coarser hotspots).")
    min_samples = st.slider("DBSCAN min_samples", 5, 100, 15, 5,
                            help="Minimum accidents to form a hotspot cluster.")

    st.header("BMSSP Routing")
    bound_B = st.slider("BMSSP cutoff bound (meters)", 5000, 30000, 10000, 1000,
                        help="Maximum route cost before considering destination unreachable.")

    run_btn = st.button("Compute Safest Route")

st.markdown("---")

if run_btn:
    with st.spinner("Geocoding areas..."):
        origin_lat, origin_lon = geocode_location_cached(origin_area)
        dest_lat, dest_lon = geocode_location_cached(dest_area)

    if origin_lat is None or origin_lon is None:
        st.error(f"Could not find coordinates for origin area: '{origin_area}'. Please refine.")
    elif dest_lat is None or dest_lon is None:
        st.error(f"Could not find coordinates for destination area: '{dest_area}'. Please refine.")
    else:
        origin_weather = None
        dest_weather = None
        origin_weather_factor = 1.0
        dest_weather_factor = 1.0

        try:
            origin_weather = get_current_weather(origin_lat, origin_lon)
            origin_weather_factor = origin_weather.get("risk_factor", 1.0)
        except Exception as e:
            origin_weather = f"Could not fetch origin weather: {e}"

        try:
            dest_weather = get_current_weather(dest_lat, dest_lon)
            dest_weather_factor = dest_weather.get("risk_factor", 1.0)
        except Exception as e:
            dest_weather = f"Could not fetch destination weather: {e}"

        if origin_weather:
            if isinstance(origin_weather, dict):
                st.markdown(
                    f"**Current Weather at Origin ({origin_area}) **  \n"
                    f"- {origin_weather['description']}, {origin_weather['temperature_c']:.1f} °C  \n"
                    f"- Humidity: {origin_weather['humidity_percent']}%  \n"
                    f"- Wind Speed: {origin_weather['wind_speed_mps']:.1f} m/s"
                )
            else:
                st.warning(origin_weather)

        if dest_weather:
            if isinstance(dest_weather, dict):
                st.markdown(
                    f"**Current Weather at Destination ({dest_area}) **  \n"
                    f"- {dest_weather['description']}, {dest_weather['temperature_c']:.1f} °C  \n"
                    f"- Humidity: {dest_weather['humidity_percent']}%  \n"
                    f"- Wind Speed: {dest_weather['wind_speed_mps']:.1f} m/s"
                )
            else:
                st.warning(dest_weather)

        try:
            with st.spinner("Loading accident data..."):
                df = load_accident_data_cached(DATA_PATH)
            if recompute or not HOTSPOTS_PATH.exists():
                with st.spinner("Detecting hotspots..."):
                    clusters, hotspot_time = detect_hotspots_cached(df, eps_m, min_samples)
                    clusters.to_csv(HOTSPOTS_PATH, index=False)
                    st.info(f"Hotspot detection took {hotspot_time:.2f} seconds")
            else:
                clusters = pd.read_csv(HOTSPOTS_PATH)
                hotspot_time = 0.0

            with st.spinner("Building graph..."):
                G, graph_build_time = build_graph_cached("Bengaluru, Karnataka, India")
                st.info(f"Graph building took {graph_build_time:.2f} seconds")

            with st.spinner("Attaching risk scores..."):
                G, risk_attach_time = attach_risk_cached(
                    G, clusters,
                    origin_coords=(origin_lat, origin_lon),
                    dest_coords=(dest_lat, dest_lon),
                    origin_weather_factor=origin_weather_factor,
                    dest_weather_factor=dest_weather_factor
                )
                st.info(f"Risk attachment took {risk_attach_time:.2f} seconds")

            route_colors = {
                "Balanced": "yellow",
                "Shortest Route (High Risk)": "red",
                "Safest Route (Longer)": "green",
                "Custom": "blue"
            }
            selected_color = route_colors.get(route_profile, "green")

            with st.spinner("Computing route and rendering map..."):
                start_route = time.perf_counter()
                path = safest_path_bmssp(G, (origin_lat, origin_lon), (dest_lat, dest_lon),
                                        alpha=alpha_val, beta=beta_val, bound_B=bound_B)
                save_map_path = MAP_PATH
                folium_map_with_route(G, path, clusters, accidents_df=df, save_path=save_map_path, route_color=selected_color)
                route_time = time.perf_counter() - start_route

            st.success(f"Done! Accidents used: {len(df):,} • Hotspots: {len(clusters)}")
            st.info(f"Route computation took {route_time:.2f} seconds")
            st.caption(f"Map file: {save_map_path}")

            if save_map_path.exists():
                html = save_map_path.read_text(encoding="utf-8")
                st.components.v1.html(html, height=680, scrolling=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Fill inputs on the left and click **Compute Safest Route**.")

st.markdown("---")
st.write("**Tip:** Choose a route profile or manually tune alpha (distance importance) and beta (risk aversion) for your preference.")
