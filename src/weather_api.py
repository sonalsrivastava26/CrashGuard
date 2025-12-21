import requests
import os

OPENWEATHERMAP_API_KEY = "8fea02f7a9118b920735aacb81e307bc"
# Mapping main weather conditions to risk multipliers
WEATHER_RISK_MAP = {
    "Clear": 1.0,
    "Clouds": 1.1,
    "Mist": 1.3,
    "Fog": 1.5,
    "Rain": 1.5,
    "Drizzle": 1.2,
    "Thunderstorm": 2.0,
    "Snow": 1.7,
    "Haze": 1.4,
}

def get_current_weather(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "metric"
    }
    response = requests.get(url, params=params, timeout=10)
    if response.status_code == 200:
        data = response.json()
        main_condition = data["weather"][0]["main"]
        risk_factor = WEATHER_RISK_MAP.get(main_condition, 1.0)
        return {
            "description": data["weather"][0]["description"].title(),
            "main_condition": main_condition,
            "risk_factor": risk_factor,
            "temperature_c": data["main"]["temp"],
            "humidity_percent": data["main"]["humidity"],
            "wind_speed_mps": data["wind"]["speed"],
            "city": data.get("name", "Unknown")
        }
    else:
        raise Exception(f"Weather API error: {response.status_code} {response.text}")
