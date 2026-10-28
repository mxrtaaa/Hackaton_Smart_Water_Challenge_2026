from __future__ import annotations
import json
import math
import re
import sys
from datetime import timedelta, timezone

from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import Booster

from business_metrics import (
    CENA_ENERGII,
    calculate_excess,
    run_business_analysis,
    summarize_rain_events,
    classify_weather_periods,
    calculate_dry_baseline,
    calculate_calendar_baseline,
    calculate_anomaly_cost,

)

from src.data_loader import (
    load_processed_chorzow,
    load_processed_weteranow,
    chorzow_processed_paths as processed_paths,
    weteranow_processed_paths,
    load_processed_rybnik,
    rybnik_processed_paths,
)
from weather_loader import load_weather_forecast, forecast_start

FIXED_TZ = timezone(timedelta(hours=1))

STATIONS = [
    {"id": "chorzow", "name": "Chorzów", "code": "CH-01", "prefix": "chorzow"},
    {"id": "ruda-weteranow", "name": "Ruda Śląska · Weteranów", "code": "RS-01", "prefix": "ruda"},
    {"id": "rybnik_pz1", "name": "Rybnik · PZ1", "code": "RY-PZ1", "prefix": "rybnik", "object": "pz1"},
    {"id": "rybnik_pnb1", "name": "Rybnik · PNB1", "code": "RY-PNB1", "prefix": "rybnik", "object": "pnb1"},
]

FLOW_MODELS = ROOT / "models" / "Predykcja przepompowanych ścieków" / "wyeksportowane modele"
ENERGY_MODELS = ROOT / "models" / "Predykcja zużycia energii" / "wyeksportowane modele"
FINAL_MODELS = ROOT / "models" / "final" / "wyeksportowane modele"


def weteranow_paths(root):
    root = Path(root)
    flow = root / "models" / "Predykcja przepompowanych ścieków" / "wyeksportowane modele"
    energy = root / "models" / "Predykcja zużycia energii" / "wyeksportowane modele"
    return {**weteranow_processed_paths(root),
            "flow_median": flow / "ruda_przepompowanie_median.cbm",
            "flow_q97": flow / "ruda_przepompowanie_q97.cbm",
            "energy_median": energy / "ruda_energia_median.txt",
            "energy_q97": energy / "ruda_energia_q97.txt"}


def require_weteranow(root):
    paths = weteranow_paths(root)
    missing = [str(p.relative_to(root)) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Brak zatwierdzonych plików Weteranów: " + ", ".join(missing))
    return paths

def station_config(station_id):
    return next((s for s in STATIONS if s["id"] == station_id), None)

def asset_paths(station):

    if "object" in station:
        obj = station["object"]
        return [rybnik_processed_paths(ROOT, obj)["flow"]] + [
            FINAL_MODELS / f"rybnik_przepompowanie_{obj}_{kind}.cbm" for kind in ("median", "q97")
        ] + [FINAL_MODELS / f"rybnik_energia_{obj}_{kind}.txt" for kind in ("median", "q97")]
    if station["id"] == "ruda-weteranow":
        paths = weteranow_paths(ROOT)
        return [paths[key] for key in ("hourly", "flow_median", "flow_q97", "energy_median", "energy_q97")]

    prefix = station["prefix"]

    return [ROOT / "reports" / station["id"] / "godziny.parquet"] + [
        FLOW_MODELS / f"{prefix}_przepompowanie_{kind}.cbm" for kind in ("median", "q97")
    ] + [ENERGY_MODELS / f"{prefix}_energia_{kind}.txt" for kind in ("median", "q97")]

def fingerprint(station_id):
    station = station_config(station_id)

    if station is None:
        raise ValueError("Nieznany obiekt.")

    paths = asset_paths(station) + [ROOT / "src" / "business_metrics.py"]

    if station_id == "chorzow":
        paths = processed_paths(ROOT) + asset_paths(station)[1:] + [ROOT / "src" / "business_metrics.py"]
    elif station_id == "ruda-weteranow":
        paths = list(require_weteranow(ROOT).values()) + [ROOT / "src" / "business_metrics.py"]
    elif "object" in station:
        paths = list(rybnik_processed_paths(ROOT, station["object"]).values()) + asset_paths(station)[1:] + [ROOT / "src" / "business_metrics.py"]

    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)

def load_hourly(report_path):
    frame = pd.read_parquet(report_path, columns=["przepompowano_m3", "energia_kwh", "opad_mm"])

    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("Raport musi mieć DatetimeIndex ze strefą czasową.")

    if frame.index.has_duplicates:
        raise ValueError("Raport zawiera zduplikowane daty.")

    frame = frame.sort_index().asfreq("h")
    frame.index = frame.index.tz_convert(FIXED_TZ)
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    frame[["przepompowano_m3", "energia_kwh"]] = frame[["przepompowano_m3", "energia_kwh"]].where(frame[["przepompowano_m3", "energia_kwh"]] >= 0)

    last = frame[["przepompowano_m3", "energia_kwh"]].dropna(how="all").index
    if last.empty:
        raise ValueError("Raport nie zawiera pomiarów przepompowania ani energii.")

    return frame.loc[:last[-1]]

def flow_features(frame, station_id=None):
    index = frame.index.tz_convert(FIXED_TZ)

    features = pd.DataFrame(index=frame.index)
    features["hour"] = index.hour
    features["dayofweek"] = index.dayofweek
    features["month"] = index.month
    features["weekend"] = (index.dayofweek >= 5).astype(int)
    features["month_sin"] = np.sin(2 * np.pi * (index.month - 1) / 12)
    features["month_cos"] = np.cos(2 * np.pi * (index.month - 1) / 12)

    rain = frame["opad_mm"]
    features["opad_0h"] = rain

    for hours in (2, 3, 6, 12, 24, 48):
        minimum = 1 if station_id in ("rybnik_pz1", "rybnik_pnb1") else hours
        features[f"opad_{hours}h"] = rain.rolling(hours, min_periods=minimum).sum()

    features["opad_max6h"] = rain.rolling(6, min_periods=1 if station_id in ("rybnik_pz1", "rybnik_pnb1") else 6).max()

    for hours in (1, 2, 3, 6, 12, 24):
        features[f"opad_lag_{hours}h"] = rain.shift(hours)

    return features

def energy_features(frame, station_id):

    # Chorzów preparation uses local calendar; Ruda uses fixed UTC+1.
    index = frame.index.tz_convert("Europe/Warsaw" if station_id == "chorzow" else FIXED_TZ)
    features = pd.DataFrame(index=frame.index)
    features["hour"], features["dayofweek"], features["month"] = index.hour, index.dayofweek, index.month

    rain = frame["opad_mm"]

    for hours in (1, 6, 12, 24, 48, 168):
        features[f"opad_{hours}h"] = rain.shift(1).rolling(hours, min_periods=hours).sum()

    last_rain = frame.index.to_series().where(rain.shift(1) > 0.1).ffill()
    features["h_od_deszczu"] = ((frame.index.to_series() - last_rain).dt.total_seconds() / 3600).clip(upper=240)

    # These energy models were trained using observed pumping as a feature.
    # This is a conditional historical prediction, not a free-running forecast.
    features["przepompowano"] = frame["przepompowano_m3"]

    return features

def infer(model, features, feature_names):

    missing = set(feature_names) - set(features.columns)

    if missing:
        raise ValueError(f"Brak implementacji cech modelu: {sorted(missing)}")

    selected = features.loc[:, feature_names]
    valid = selected.notna().all(axis=1)
    predictions = pd.Series(np.nan, index=selected.index, dtype=float)

    if valid.any():
        predictions.loc[valid] = model.predict(selected.loc[valid])

    return predictions

@lru_cache(maxsize=4)

def exported_models(station_id, signature):

    paths = asset_paths(station_config(station_id))
    result = []

    for column, path in zip(("prognoza", "q97", "energia_prognoza", "energia_q97"), paths[1:]):
        if path.suffix == ".cbm":
            model = CatBoostRegressor()
            model.load_model(str(path))
            names = model.feature_names_

        else:
            model = Booster(model_str=path.read_text(encoding="utf-8"))
            names = model.feature_name()

        result.append((column, model, names, path))

    return result

@lru_cache(maxsize=4)

def load_station(station_id, signature):

    station = station_config(station_id)
    report, median_path, q97_path, energy_path, energy_q97_path = asset_paths(station)
    if station_id == "chorzow":
        frame = load_processed_chorzow(ROOT)[0]
    elif station_id == "ruda-weteranow":
        frame = load_processed_weteranow(ROOT)[0]
    else:
        frame = load_processed_rybnik(ROOT, station["object"])[0]

    analysis = run_business_analysis(frame, price=CENA_ENERGII)
    detail = analysis["detail"].copy()
    flow = flow_features(frame, station_id)
    energy = energy_features(frame, station_id) if station_id == "chorzow" else None

    models = []

    for column, model, names, path in exported_models(station_id, signature):
        if column.startswith("energia") and energy is None:

            # Weteranów history and forecast both use predicted pumping, as requested.
            energy_inputs = frame.copy()
            energy_inputs["przepompowano_m3"] = detail["prognoza"]
            energy = energy_features(energy_inputs, station_id)

        detail[column] = infer(model, energy if column.startswith("energia") else flow, names)
        models.append({"output": column, "path": str(path.relative_to(ROOT)), "features": names})

    # Cost references are fitted on the full archive, before selecting 30 days.
    calendar = calculate_calendar_baseline(frame)
    detail["baseline_kalendarzowy_m3"] = calendar["bazowe_m3"]
    detail["baseline_kalendarzowy_kwh"] = calendar["bazowe_kwh"]
    if station_id == "chorzow":
        # The cost reference must not absorb the observed pumping anomaly.
        # Keep the existing conditional historical energy chart unchanged.
        weather_inputs = frame.copy()
        weather_inputs["przepompowano_m3"] = detail["prognoza"]
        weather_features = energy_features(weather_inputs, station_id)
        _, model, names, _ = next(m for m in exported_models(station_id, signature)
                                  if m[0] == "energia_prognoza")
        weather_energy = infer(model, weather_features, names)
    else:
        weather_energy = detail["energia_prognoza"]
    detail["baseline_pogodowy_kwh"] = weather_energy.where(weather_energy >= 0)
    detail["anomaly"] = detail["przepompowano_m3"].notna() & detail["q97"].notna() & (detail["przepompowano_m3"] > detail["q97"])

    return detail, models

def clean(value):
    if value is pd.NA or value is pd.NaT or value is None:
        return None

    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None

    return value

def records(frame, index_name=None):

    result = frame.copy()
    if index_name:
        result = result.rename_axis(index_name).reset_index()

    return clean(result.to_dict("records"))

def window_detail(detail, days):
    return detail.loc[detail.index > detail.index[-1] - pd.Timedelta(days=days)].copy()

def summarize(detail, price):
    detail = calculate_excess(detail, price_per_kwh=price)
    episodes = summarize_rain_events(detail, price_per_kwh=price)

    wet = detail.loc[detail["okres"].eq("mokry")].copy()

    wet["miesiac"] = wet.index.tz_convert("Europe/Warsaw").strftime("%Y-%m")

    monthly = wet.groupby("miesiac").agg(

        godziny_mokre=("okres", "size"), godziny_wycenione=("nadwyzka_kwh", "count"),
        nadwyzka_m3=("nadwyzka_m3", lambda x: x.sum(min_count=1)),
        nadwyzka_kwh=("nadwyzka_kwh", lambda x: x.sum(min_count=1)),

    )

    monthly["pokrycie"] = monthly["godziny_wycenione"] / monthly["godziny_mokre"]
    monthly["bilans_zl"] = monthly["nadwyzka_kwh"] * price

    return {

        "episodes": records(episodes, "epizod_id"), "monthly": records(monthly, "miesiac"),
        "summary": clean({"godziny_mokre": len(wet), "godziny_wycenione": wet["nadwyzka_kwh"].count(),

            "pelne_epizody": int(episodes["kompletny"].sum()), "koszt_pelnych_epizodow_zl": episodes["koszt_zl"].sum()}),

    }

def anomalies_for(detail, price=CENA_ENERGII):

    groups = []

    for time, row in detail.iterrows():
        if not row["anomaly"]:
            continue

        if not groups or time - groups[-1][-1][0] != pd.Timedelta(hours=1):
            groups.append([])

        groups[-1].append((time, row))

    result = []

    for group in reversed(groups):

        deviation = [100 * (r["przepompowano_m3"] / r["prognoza"] - 1) for _, r in group if r["prognoza"] > 0]
        event = detail.loc[[time for time, _ in group]].copy()
        weather_cost = calculate_anomaly_cost(event.assign(
            bazowe_m3=event["prognoza"].where(event["prognoza"] >= 0),
            bazowe_kwh=event["baseline_pogodowy_kwh"],
        ), price)
        calendar_cost = calculate_anomaly_cost(event.assign(
            bazowe_m3=event["baseline_kalendarzowy_m3"],
            bazowe_kwh=event["baseline_kalendarzowy_kwh"],
        ), price)
        result.append({"id": group[0][0].isoformat(), "time": group[0][0], "end": group[-1][0], "title": "Przekroczenie Q97",

            "deviation": max(deviation) if deviation else None,
            "ponad_q97": sum(r["przepompowano_m3"] - r["q97"] for _, r in group),
            "koszt_pogodowy_zl": weather_cost["koszt_zl"],
            "koszt_kalendarzowy_zl": calendar_cost["koszt_zl"],
            "wycena_pogodowa": weather_cost, "wycena_kalendarzowa": calendar_cost,
            "description": "Przekroczenie kwantyla z wyeksportowanego modelu CatBoost. Historyczne porównanie z pomiarem nie potwierdza awarii ani możliwości uzyskania oszczędności."})

    return clean(result)

def predict_station_future(history, weather, models, station_id):

    info = dict(weather.attrs)

    start, end = pd.Timestamp(info["start"]), pd.Timestamp(info["endExclusive"])

    if start <= history.index[-1]:
        raise ValueError("Prognoza musi zaczynać się po ostatnim pomiarze historii.")

    rain = weather.set_index("Data_czas")["suma_opadu_mm"].tz_convert(FIXED_TZ)
    inputs = pd.DataFrame({"opad_mm": rain, "przepompowano_m3": np.nan, "energia_kwh": np.nan})
    outputs = pd.DataFrame(index=inputs.index)

    flow = flow_features(inputs, station_id)

    for column, model, names, _ in models:

        if not column.startswith("energia"):
            outputs[column] = infer(model, flow, names)


    predicted_features = inputs.copy()
    predicted_features["przepompowano_m3"] = outputs["prognoza"]

    energy = energy_features(predicted_features, station_id)

    for column, model, names, _ in models:
        if column.startswith("energia"):
            outputs[column] = infer(model, energy, names)

    outputs = outputs.where(outputs >= 0)
    target = pd.date_range(start, end, freq="h", inclusive="left").tz_convert(FIXED_TZ)
    public = outputs.reindex(target).copy()

    public["przepompowano_m3"] = np.nan
    public["energia_kwh"] = np.nan
    public["opad_mm"] = rain.reindex(target)
    public["anomaly"] = False
    public["data_kind"] = "prediction"

    public["weather_source"] = f"Open-Meteo EC46 · scenariusz {info.get('ensembleMember', 0)}"

    # Warm-up keeps the 48h weather classification. Missing days remain missing.
    actual = history[["przepompowano_m3", "energia_kwh", "opad_mm"]]

    context = pd.concat([actual, inputs.loc[inputs.index > actual.index[-1]]]).sort_index()
    baseline = calculate_dry_baseline(classify_weather_periods(context))
    projected = baseline.reindex(target).copy()
    projected["przepompowano_m3"] = public["prognoza"]
    projected["energia_kwh"] = public["energia_prognoza"]
    projected["data_kind"] = "prediction"

    projected = calculate_excess(projected, CENA_ENERGII)

    for field in ("nadwyzka_m3", "nadwyzka_kwh", "koszt_nadwyzki_zl"):
        public["predicted_" + field] = projected[field]

    for field in ("okres", "bazowe_m3", "bazowe_kwh"):
        public[field] = projected[field]

    count = int(public["energia_prognoza"].count())

    info.update({"status": "ok" if count == len(public) else "partial",
                 "predictedEnergyHours": count, "predictedPumpingHours": int(public["prognoza"].count()),
                 "expectedHours": len(public)})

    return public, projected, info

def predict_chorzow_future(history, weather, models):
    return predict_station_future(history, weather, models, "chorzow")

@lru_cache(maxsize=4)

def _chorzow_forecast(signature, day, bucket):

    detail, _ = load_station("chorzow", signature)
    weather = load_weather_forecast(50.292298, 18.954600, cache_path=ROOT / "ui" / ".runtime" / "chorzow-weather.json")

    return predict_chorzow_future(detail, weather, exported_models("chorzow", signature))

def chorzow_forecast():

    now = pd.Timestamp.now(tz="UTC")
    return _chorzow_forecast(fingerprint("chorzow"), forecast_start(now).isoformat(), now.floor("6h").isoformat())

@lru_cache(maxsize=4)

def _weteranow_forecast(signature, day, bucket):

    detail, _ = load_station("ruda-weteranow", signature)
    weather = load_weather_forecast(50.261259, 18.894069, cache_path=ROOT / "ui" / ".runtime" / "weteranow-weather.json")

    return predict_station_future(detail, weather, exported_models("ruda-weteranow", signature), "ruda-weteranow")

def weteranow_forecast():

    now = pd.Timestamp.now(tz="UTC")
    return _weteranow_forecast(fingerprint("ruda-weteranow"), forecast_start(now).isoformat(), now.floor("6h").isoformat())


@lru_cache(maxsize=4)
def _rybnik_forecast(station_id, signature, day, bucket):
    detail, _ = load_station(station_id, signature)
    # Existing Rybnik weather coordinates from src/weather_loader.py.
    # Both objects use this weather point; there are no separate coordinates in the project.
    weather = load_weather_forecast(50.114718, 18.533368,
                                   cache_path=ROOT / "ui" / ".runtime" / "rybnik-weather.json")
    return predict_station_future(detail, weather, exported_models(station_id, signature), station_id)


def rybnik_forecast(station_id):
    if station_id not in ("rybnik_pz1", "rybnik_pnb1"):
        raise ValueError("Nieznany obiekt Rybnika.")
    now = pd.Timestamp.now(tz="UTC")
    return _rybnik_forecast(station_id, fingerprint(station_id), forecast_start(now).isoformat(), now.floor("6h").isoformat())


def station_forecast(station_id):
    if station_id == "chorzow":
        return chorzow_forecast()
    if station_id == "ruda-weteranow":
        return weteranow_forecast()
    return rybnik_forecast(station_id)

def energy_totals(frame, price, *, kind="measurement", metric="energy"):

    energy = frame["energia_kwh"]
    daily = []

    for day, values in energy.groupby(energy.index.tz_convert("Europe/Warsaw").strftime("%Y-%m-%d")):
        daily.append({"date": day, "energyKwh": values.sum(min_count=1), "costZl": values.sum(min_count=1) * price,
                      "pricedHours": int(values.count()), "expectedHours": len(values), "dataKind": kind})

    return clean({"dataKind": kind, "metric": metric, "energyKwh": energy.sum(min_count=1), "daily": daily,
                  "totalCostZl": energy.sum(min_count=1) * price,
                  "pricedHours": int(energy.count()), "expectedHours": len(frame),
                  "complete": bool(energy.notna().all()),
                  "start": frame.index[0] if len(frame) else None, "end": frame.index[-1] if len(frame) else None})

def dashboard(station_id):

    detail, models = load_station(station_id, fingerprint(station_id))

    visible = window_detail(detail, 30)
    station = station_config(station_id)
    result = {"schemaVersion": 2, "source": "reports-models", "businessEngine": "python", "asOf": detail.index[-1].isoformat(),

        "station": {k: station[k] for k in ("id", "name", "code")},
        "stations": [{k: s[k] for k in ("id", "name", "code")} for s in STATIONS],
        "pricePerKwh": CENA_ENERGII, "history": records(visible, "time"), "forecast": [],
        "anomalies": anomalies_for(visible),
        "anomalyCostMethod": {
            "pricePerKwh": CENA_ENERGII,
            "weatherBaseline": "Model energii z opadami, kalendarzem i predykcją pompowania; bez pomiaru pompowania jako wejścia.",
            "calendarBaseline": "Mediany wszystkich kompletnych historycznych godzin: miesiąc+godzina+weekend, następnie fallback z business_metrics.py; minimum 5 próbek, bez opadów.",
            "formula": "max(0, suma(pomiar kWh - baseline kWh)) × cena; tylko godziny danego przekroczenia Q97, wszystkie wymagane do wyceny.",
        },
        "businessByRange": {str(days): summarize(window_detail(detail, days), CENA_ENERGII) for days in (7, 30)},
        "provenance": {"report": str(asset_paths(station)[0].relative_to(ROOT)), "models": models,
            "businessModule": "src/business_metrics.py", "forecastUnavailable": "Brak przyszłych cech pogodowych; modele nie wyznaczają samodzielnie prognozy na 30 dni.",
            "note": "Archiwalne pomiary z raportów projektu i rzeczywista inferencja zapisanych modeli. Porównanie historyczne; nie jest niezależną oceną jakości ani monitoringiem na żywo."}}

    if station_id == "chorzow":
        _, raw_info = load_processed_chorzow(ROOT)

        result["source"] = "processed-models"
        result["stations"] = [result["station"]]
        result["historyRanges"] = [7, 14, 30]
        result["businessByRange"]["14"] = summarize(window_detail(detail, 14), CENA_ENERGII)
        result["historyTotals"] = {str(days): energy_totals(window_detail(detail, days), CENA_ENERGII) for days in (7, 14, 30)}

        for days in (7, 14, 30):

            window = window_detail(detail, days)

            result["historyTotals"][str(days)]["pumpingM3"] = clean(window["przepompowano_m3"].sum(min_count=1))
            result["historyTotals"][str(days)]["pumpingHours"] = int(window["przepompowano_m3"].count())

        result["provenance"].pop("report", None)
        result["provenance"].update(raw_info)
        result["provenance"]["note"] = "Rzeczywiste przyrosty liczników z data/processed/Chorzów; zakresy wstecz od końca archiwum. Prognoza ma odrębne przyszłe daty."

        try:

            future, projected, weather_info = chorzow_forecast()
            result["forecast"] = records(future, "time")
            result["weatherForecast"] = weather_info
            result["forecastTotals"] = energy_totals(projected, CENA_ENERGII, kind="prediction")
            result["forecastTotals"]["pumpingM3"] = clean(future["prognoza"].sum(min_count=1))
            result["forecastTotals"]["pumpingHours"] = int(future["prognoza"].count())
            result["provenance"].pop("forecastUnavailable", None)

        except Exception as error:

            result["weatherForecast"] = {"status": "unavailable", "error": str(error)}
            result["provenance"]["forecastUnavailable"] = str(error)

    elif station_id == "ruda-weteranow":

        _, source_info = load_processed_weteranow(ROOT)
        result["source"] = "weteranow-models"

        result["historyRanges"] = [7, 14, 30]
        result["businessByRange"]["14"] = summarize(window_detail(detail, 14), CENA_ENERGII)
        result["provenance"].pop("report", None)
        result["provenance"].update(source_info)
        result["provenance"]["energyMethod"] = "LightGBM Weteranów z cechą przepompowano = predykcja CatBoost Weteranów; także w porównaniu historycznym. Pozostałe cechy: kalendarz i opady zgodnie z treningiem."
        result["provenance"]["assetApproval"] = "Użytkownik dopuścił dokładnie wskazane istniejące nazwy plików mimo braku prefiksu 'ruda śląska ... weteranów'."
        result["provenance"]["note"] = "Wyłącznie zatwierdzone pliki Weteranów z models i data/processed/Ruda Śląska. Historia liczona wstecz od końca pełnych pomiarów."

        try:

            future, projected, weather_info = weteranow_forecast()
            result["forecast"] = records(future, "time")
            result["weatherForecast"] = weather_info
            result["forecastTotals"] = energy_totals(projected, CENA_ENERGII, kind="prediction")
            result["forecastTotals"].update({"pumpingM3": clean(future["prognoza"].sum(min_count=1)), "pumpingHours": int(future["prognoza"].count())})
            result["provenance"].pop("forecastUnavailable", None)

        except Exception as error:
            result["weatherForecast"] = {"status": "unavailable", "error": str(error)}
            result["provenance"]["forecastUnavailable"] = str(error)

    elif "object" in station:
        _, source_info = load_processed_rybnik(ROOT, station["object"])
        result["source"] = "rybnik-final-models"
        result["historyRanges"] = [7, 14, 30]
        result["businessByRange"]["14"] = summarize(window_detail(detail, 14), CENA_ENERGII)
        result["provenance"].pop("report", None)
        result["provenance"].update(source_info)
        result["provenance"]["energyMethod"] = "LightGBM obiektu z predykcją pompowania CatBoost i opadami; oddzielny pomiar kWh z raportu."
        result["provenance"]["weatherCoordinates"] = {"latitude": 50.114718, "longitude": 18.533368,
                                                        "source": "Punkt Rybnika z src/weather_loader.py, wspólny dla PZ1 i PNB1."}
        result["historyTotals"] = {}
        for days in (7, 14, 30):
            window = window_detail(detail, days)
            result["historyTotals"][str(days)] = energy_totals(window, CENA_ENERGII)
            result["historyTotals"][str(days)].update({
                "pumpingM3": clean(window["przepompowano_m3"].sum(min_count=1)),
                "pumpingHours": int(window["przepompowano_m3"].count())})
        try:
            future, projected, weather_info = rybnik_forecast(station_id)
            result["forecast"] = records(future, "time")
            result["weatherForecast"] = weather_info
            result["forecastTotals"] = energy_totals(projected, CENA_ENERGII, kind="prediction")
            result["forecastTotals"].update({"pumpingM3": clean(future["prognoza"].sum(min_count=1)),
                                             "pumpingHours": int(future["prognoza"].count())})
            result["provenance"].pop("forecastUnavailable", None)
        except Exception as error:
            result["weatherForecast"] = {"status": "unavailable", "error": str(error)}
            result["provenance"]["forecastUnavailable"] = str(error)
    return result

def select_cost_dates(frame, start_date=None, end_date=None):

    if start_date is None and end_date is None:
        return frame

    if not all(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) for value in (start_date, end_date)):
        raise ValueError("Podaj obie daty wyceny w formacie RRRR-MM-DD.")

    start = pd.Timestamp(start_date).tz_localize("Europe/Warsaw")
    end = pd.Timestamp(end_date).tz_localize("Europe/Warsaw")

    if start > end:
        raise ValueError("Data początkowa musi poprzedzać końcową.")

    first_day = frame.index[0].tz_convert("Europe/Warsaw").normalize()
    last_day = frame.index[-1].tz_convert("Europe/Warsaw").normalize()

    if start < first_day or end > last_day:
        raise ValueError("Daty muszą mieścić się w wybranym okresie historii lub prognozy.")

    selected = frame.loc[(frame.index >= start) & (frame.index < end + pd.DateOffset(days=1))]

    if selected.empty:
        raise ValueError("Brak godzin w wybranym zakresie dat.")

    return selected.copy()

def cost_scenarios(station_id, days, prices, period="history", metric="wet_excess", start_date=None, end_date=None, energy_basis="measured"):

    if days not in (7, 14, 30):
        raise ValueError("Nieobsługiwany zakres historii.")

    if period not in ("history", "forecast") or metric not in ("wet_excess", "energy"):
        raise ValueError("Nieznany okres lub podstawa wyceny.")

    values = [prices.get(key) for key in ("lowPrice", "basePrice", "highPrice")]

    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Uzupełnij każdą cenę nieujemną liczbą.")

    if values != sorted(values):
        raise ValueError("Zachowaj kolejność cen: dolna ≤ bazowa ≤ górna.")

    detail, _ = load_station(station_id, fingerprint(station_id))

    if period == "forecast":
        if days != 30:
            raise ValueError("Prognoza dotyczy 30 dni.")
        _, visible, _ = station_forecast(station_id)

    else:
        visible = window_detail(detail, days)

    if energy_basis not in ("measured", "predicted"):
        raise ValueError("Nieznana podstawa zużycia energii.")

    kind = "prediction" if period == "forecast" else "measurement"

    visible = select_cost_dates(visible, start_date, end_date)
    if period == "history" and energy_basis == "predicted":
        visible = visible.copy()
        visible["energia_kwh"] = visible["energia_prognoza"]
        visible["przepompowano_m3"] = visible["prognoza"]
        kind = "prediction"

    result = []

    for name, price in zip(("Dolny", "Bazowy", "Górny"), values):
        analysis = summarize(visible, price)
        valuation = energy_totals(visible, price, kind=kind, metric=metric)
        valuation["pumpingM3"] = clean(visible["przepompowano_m3"].sum(min_count=1))
        valuation["pumpingHours"] = int(visible["przepompowano_m3"].count())

        if metric == "wet_excess":
            valuation.update({"totalCostZl": analysis["summary"]["koszt_pelnych_epizodow_zl"] if analysis["summary"]["godziny_wycenione"] else None,
                              "pricedHours": analysis["summary"]["godziny_wycenione"],
                              "expectedHours": analysis["summary"]["godziny_mokre"],
                              "complete": all(e["kompletny"] for e in analysis["episodes"])})

        result.append({"name": name, "price": price, **analysis, "valuation": valuation})

    return result

def handle(request):

    station_id = request.get("station", "chorzow")
    if request.get("action") == "costs":
        return cost_scenarios(station_id, request.get("days"), request.get("prices", {}), request.get("period", "history"), request.get("metric", "wet_excess"), request.get("startDate"), request.get("endDate"), request.get("energyBasis", "measured"))

    if request.get("action", "dashboard") != "dashboard":
        raise ValueError("Nieznane działanie backendu.")

    return dashboard(station_id)

if __name__ == "__main__":
    for line in sys.stdin:
        try:
            request = json.loads(line)
            result = {"id": request.get("id"), "result": handle(request)}

        except Exception as error:
            result = {"id": locals().get("request", {}).get("id"), "error": f"{type(error).__name__}: {error}"}

        print(json.dumps(clean(result), ensure_ascii=False, allow_nan=False), flush=True)
