import logging
import time

import requests

import config
import network_model
import weather.openmeteo_collate as openmeteo_collate


def __placeholder_weather():
    return [10, 10, 100, 0, 0, 270, 3, 22, 1018, 4]  # Placeholder data


retry_count = 3
retry_delay = 2  # seconds


def get_current_weather_at(tiploc, source="openmeteo"):
    node = network_model.G.nodes(data=True)[tiploc]
    if "latlong" in node:
        if source == "openweather":
            return _get_openweather_forecast(node, tiploc)
        elif source == "openmeteo":
            current, forecast = openmeteo_collate.current_and_forecast(node["latlong"])
            return current
        else:
            logging.error(f"Unknown weather source: {source}, defaulting to placeholder data.")
            return __placeholder_weather()
    else:
        print(f"Latitude and longitude not found for {tiploc}")
        return __placeholder_weather()


def _get_openweather_forecast(node, tiploc):
    # Parameters to include in the API call
    params = {
        "lat": node["latlong"][0],
        "lon": node["latlong"][1],
        "appid": config.openweather_API_key,
        "units": "metric",
        "exclude": "minutely,hourly,daily,alerts"
        # Excluding data not needed for this function to save data and time
    }
    for i in range(retry_count):
        response = requests.get(config.openweather_base_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            return _process_openweather_response(data)
        else:
            time.sleep(retry_delay)

        logging.info(
            f"Failed to retrieve weather data for {tiploc} attempt {i} of {retry_count}. Response: " + response.text)
    return __placeholder_weather()


def _process_openweather_response(data):
    current = data['current']
    temperature = current['temp']
    dew_point = current['dew_point']
    relative_humidity = current['humidity']
    precipitation = current.get('rain', {}).get('1h', 0) if 'rain' in current else 0
    snow = current.get('snow', {}).get('1h', 0) if 'snow' in current else 0
    wind_direction = current['wind_deg']
    wind_speed = current['wind_speed']
    peak_wind_gust = current.get('wind_gust', 0)
    pressure = current['pressure']
    # OpenWeather does not provide direct sunlight duration in current data, so this is left as None
    total_sun = None
    cloud_cover = current['clouds']
    return [
        temperature, dew_point, relative_humidity,
        precipitation, snow, wind_direction, wind_speed,
        peak_wind_gust, pressure, cloud_cover
    ]

