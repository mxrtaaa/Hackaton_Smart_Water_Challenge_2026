import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from pathlib import Path


def load_weather_data(export_path, latitude, longitude, start_date, end_date, hourly):

    # Konfiguracja klienta z bezterminowym cachem (dane historyczne się nie zmieniają)

    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly,
        "timezone": "Europe/Warsaw"
    }


    # Wywołanie API

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]


    # Przetworzenie struktur C++ na DataFrame

    hourly = response.Hourly()
    hourly_precipitation = hourly.Variables(0).ValuesAsNumpy()

    hourly_data = {"Data_czas": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}
    hourly_data["suma_opadu_mm"] = hourly_precipitation

    df_pogoda = pd.DataFrame(data=hourly_data)


    # Konwersja czasu na lokalny i usunięcie znacznika strefy (tz-naive)

    df_pogoda['Data_czas'] = df_pogoda['Data_czas'].dt.tz_convert('Europe/Warsaw').dt.tz_localize(None)

    sciezka_wyjsciowa = export_path
    Path(sciezka_wyjsciowa).parent.mkdir(parents=True, exist_ok=True)
    df_pogoda.to_parquet(sciezka_wyjsciowa)
if __name__ == "__main__":

    # Ruda Śląska
    load_weather_data('../data/processed/Ruda Śląska/suma_opadow.parquet',
                      latitude='50.261259',
                      longitude='18.894069',
                      start_date='2024-01-01',
                      end_date='2026-10-24',
                      hourly='precipitation'
                      )


    # Chorzów

    load_weather_data('../data/processed/Chorzów/suma_opadow.parquet',
                      latitude='50.292298',
                      longitude='18.954600',
                      start_date='2024-01-01',
                      end_date='2026-10-24',
                      hourly='precipitation'
                      )