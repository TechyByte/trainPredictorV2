import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry
from time import sleep

# Set up the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"

def current_and_forecast(latlong):
    sleep(0.1) # Sleep to avoid rate limiting
    params = {
        "latitude": latlong[0],
        "longitude": latlong[1],
        "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "snowfall", "cloud_cover", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "snowfall", "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
        "past_days": 1,
        "timezone": "Europe/London"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity = hourly.Variables(1).ValuesAsNumpy()
    hourly_dew_point = hourly.Variables(2).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(4).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(5).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(6).ValuesAsNumpy()
    hourly_wind_speed = hourly.Variables(7).ValuesAsNumpy()
    hourly_wind_direction = hourly.Variables(8).ValuesAsNumpy()
    hourly_wind_gusts = hourly.Variables(9).ValuesAsNumpy()

    forecast = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ), "temperature_2m": hourly_temperature, "relative_humidity_2m": hourly_relative_humidity,
        "dew_point_2m": hourly_dew_point, "precipitation": hourly_precipitation, "snowfall": hourly_snowfall,
        "surface_pressure": hourly_surface_pressure, "cloud_cover": hourly_cloud_cover,
        "wind_speed_10m": hourly_wind_speed, "wind_direction_10m": hourly_wind_direction,
        "wind_gusts_10m": hourly_wind_gusts}

    forecast = pd.DataFrame(data=forecast)

    # Current values. The order of variables needs to be the same as requested.
    current = response.Current()
    current_temperature = current.Variables(0).Value()
    current_relative_humidity = current.Variables(1).Value()
    current_precipitation = current.Variables(2).Value()
    current_snowfall = current.Variables(3).Value()
    current_cloud_cover = current.Variables(4).Value()
    current_pressure = current.Variables(5).Value()
    current_wind_speed = current.Variables(6).Value()
    current_wind_direction = current.Variables(7).Value()
    current_wind_gusts = current.Variables(8).Value()
    current_dew_point = hourly_dew_point[0]

    return ([current_temperature, current_dew_point, current_relative_humidity,
        current_precipitation, current_snowfall, current_wind_direction, current_wind_speed,
        current_wind_gusts, current_pressure, current_cloud_cover], forecast)
