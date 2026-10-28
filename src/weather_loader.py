import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from pathlib import Path
from datetime import timezone, timedelta
from sciezka_projektu import get_project_root
import json
import math
from urllib.parse import urlencode
from urllib.request import urlopen

DATA_PATH = get_project_root() / "data" / "processed"

SEASONAL_URL = "https://seasonal-api.open-meteo.com/v1/seasonal"
SEASONAL_MODEL = "ecmwf_ec46"
DEFAULT_ENSEMBLE_MEMBER = 0


def _weather_json(url):
    with urlopen(url, timeout=25) as response:
        return json.load(response)


def forecast_start(now=None):
    """Start następnego dnia w Polsce; horyzont to 720 kolejnych godzin."""
    current = pd.Timestamp.now(tz="Europe/Warsaw") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        raise ValueError("Data prognozy musi mieć strefę czasową.")
    return current.tz_convert("Europe/Warsaw").normalize() + pd.DateOffset(days=1)


def load_weather_forecast(latitude, longitude, forecast_days=30, *, now=None,
                          cache_path=None, request_json=None, ensemble_member=DEFAULT_ENSEMBLE_MEMBER):

    if (
        isinstance(forecast_days, bool)
        or not isinstance(forecast_days, int)
        or not 1 <= forecast_days <= 30
    ):
        raise ValueError("Obsługiwany horyzont prognozy: 1–30 dni.")

    if (
        isinstance(ensemble_member, bool)
        or not isinstance(ensemble_member, int)
        or not 0 <= ensemble_member <= 50
    ):
        raise ValueError("Scenariusz EC46 musi być liczbą całkowitą od 0 do 50.")

    # Zakres czasowy prognozy
    current = (
        pd.Timestamp.now(tz="UTC")
        if now is None
        else pd.Timestamp(now).tz_convert("UTC")
    )

    start = forecast_start(current).tz_convert("UTC")
    end = start + pd.Timedelta(days=forecast_days)

    # Parametry Open-Meteo
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "hourly": "precipitation",
        "models": SEASONAL_MODEL,
        "forecast_days": forecast_days + 1,
        "past_days": 8,
        "temporal_resolution": "hourly",
        "timezone": "UTC",
        "timeformat": "unixtime"
    }

    # Konfiguracja cache
    cache_file = Path(cache_path) if cache_path else None

    cache_key = {
        **params,
        "target_start": start.isoformat(),
        "ensemble_member": ensemble_member
    }

    payload, fetched_at, cached = None, current, False

    if cache_file and cache_file.exists():
        try:
            saved = json.loads(
                cache_file.read_text(encoding="utf-8")
            )

            age = current - pd.Timestamp(saved["fetchedAt"])

            if (
                saved["key"] == cache_key
                and pd.Timedelta(0) <= age < pd.Timedelta(hours=6)
            ):
                payload = saved["payload"]
                fetched_at = pd.Timestamp(saved["fetchedAt"])
                cached = True

        except (ValueError, KeyError, TypeError):
            pass

    # Pobranie prognozy
    if payload is None:
        payload = (request_json or _weather_json)(
            SEASONAL_URL + "?" + urlencode(params)
        )

    if not isinstance(payload, dict) or payload.get("error"):
        raise ValueError("Open-Meteo nie zwróciło poprawnej prognozy.")

    hourly = payload.get("hourly") or {}

    variable = (
        "precipitation"
        if ensemble_member == 0
        else f"precipitation_member{ensemble_member:02d}"
    )

    times = hourly.get("time")
    rain = hourly.get(variable)

    if (
        not isinstance(times, list)
        or not times
        or not isinstance(rain, list)
        or len(times) != len(rain)
    ):
        raise ValueError("Pusta lub niekompletna odpowiedź godzinowa Open-Meteo.")

    if payload.get("hourly_units", {}).get(variable) != "mm":
        raise ValueError(
            "Nieoczekiwana jednostka opadów Open-Meteo; wymagane mm."
        )

    index = pd.DatetimeIndex(
        pd.to_datetime(times, unit="s", utc=True)
    )

    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Nieprawidłowa oś czasu prognozy Open-Meteo.")

    if any(t.minute or t.second for t in index):
        raise ValueError("Prognoza musi mieć znaczniki pełnych godzin.")

    if any(
        x is not None
        and (
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or x < 0
        )
        for x in rain
    ):
        raise ValueError("Prognoza opadów zawiera niepoprawne wartości.")

    # Przygotowanie DataFrame
    frame = pd.DataFrame(
        {"suma_opadu_mm": rain},
        index=index
    ).asfreq("h")

    # Missing timestamps remain unknown, including missing future days.
    wanted = pd.date_range(
        start - pd.Timedelta(days=7),
        end,
        freq="h",
        inclusive="left"
    )

    frame = frame.reindex(wanted)

    future = frame.loc[
        (frame.index >= start) & (frame.index < end)
    ]

    if not future["suma_opadu_mm"].notna().any():
        raise ValueError(
            "Open-Meteo nie ma opadów dla wymaganych przyszłych dat."
        )

    frame.index.name = "Data_czas"
    frame = frame.reset_index()

    # Metadane prognozy
    frame.attrs = {
        "endpoint": SEASONAL_URL,
        "model": SEASONAL_MODEL,
        "ensembleMember": ensemble_member,
        "weatherVariable": variable,
        "ensembleAggregation": "none",
        "selectionPolicy": "fixed-member",
        "retrievedAt": fetched_at.isoformat(),
        "cacheHit": cached,
        "start": start.isoformat(),
        "endExclusive": end.isoformat(),
        "expectedHours": forecast_days * 24,
        "availableWeatherHours": int(
            future["suma_opadu_mm"].count()
        ),
        "nativeResolutionHours": 6,
        "outputResolutionHours": 1,
        "note": (
            f"EC46: scenariusz {ensemble_member}, "
            "bez średniej między scenariuszami. "
            "Natywne 6 h; interpolacja godzinowa Open-Meteo. "
            "To jeden możliwy przebieg pogody, nie pomiar."
        )
    }

    # Zapis cache
    if cache_file and not cached:
        cache_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cache_file.write_text(
            json.dumps({
                "key": cache_key,
                "fetchedAt": current.isoformat(),
                "payload": payload
            }),
            encoding="utf-8"
        )

    return frame



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


    # Konwersja czasu na lokalny

    
    df_pogoda["Data_czas"] = (
        df_pogoda["Data_czas"]
        .dt.tz_convert(timezone(timedelta(hours=1)))
    )


    sciezka_wyjsciowa = export_path
    Path(sciezka_wyjsciowa).parent.mkdir(parents=True, exist_ok=True)
    df_pogoda.to_parquet(sciezka_wyjsciowa)
if __name__ == "__main__":

    # Ruda Śląska
    load_weather_data(
        DATA_PATH / "Ruda Śląska" / "suma_opadow.parquet",
        latitude='50.261259',
        longitude='18.894069',
        start_date='2024-01-01',
        end_date='2026-10-24',
        hourly='precipitation'
        )


    # Chorzów

    load_weather_data(
        DATA_PATH / "Chorzów" / "suma_opadow.parquet",
        latitude='50.292298',
        longitude='18.954600',
        start_date='2024-01-01',
        end_date='2026-10-24',
        hourly='precipitation'
        )

    # Rybnik

    load_weather_data(
        DATA_PATH / "Rybnik" / "suma_opadow.parquet",
        latitude="50.114718",
        longitude="18.533368",
        start_date="2024-01-01",
        end_date="2026-10-28",
        hourly="precipitation"
    )
