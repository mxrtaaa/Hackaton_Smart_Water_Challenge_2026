import pandas as pd
import numpy as np
from pathlib import Path
from data_preparation import convert_meter_data, set_timezone
from business_metrics import print_info, calculate_excess, calculate_dry_baseline, classify_weather_periods, summarize_rain_events


DATA_PATH = Path("./data/processed/Chorzów")

df_ch_opady = pd.read_parquet(DATA_PATH / "suma_opadow.parquet")
df_ch_przeplyw = pd.read_parquet(DATA_PATH / "przeplyw.parquet")
df_ch_energia = pd.read_parquet(DATA_PATH / "energia.parquet")
df_ch_scieki = pd.read_parquet(DATA_PATH / "scieki.parquet")

# CHORZÓW

df_ch_scieki_h = convert_meter_data(
    df_ch_scieki,
    "Ilosc_przepompowanych_sciekow",
    "przepompowano_m3"
)

df_ch_energia_h = convert_meter_data(
    df_ch_energia,
    "Zuzycie_energii_elektrycznej",
    "energia_kwh"
)

df_ch_opady = set_timezone(df_ch_opady)

df_ch_opady["suma_opadu_mm"] = pd.to_numeric(
    df_ch_opady["suma_opadu_mm"],
    errors="coerce"
)

df_ch_opady_h = (
    df_ch_opady["suma_opadu_mm"]
    .groupby(level=0).sum(min_count=1)
    .resample("1h").sum(min_count=1)
)

df_ch_opady_h.name = "opad_mm"

df_chorzow = pd.concat(
    [df_ch_scieki_h, df_ch_energia_h],
    axis=1
).sort_index()

df_chorzow = df_chorzow.join(
    df_ch_opady_h,
    how="left"
)

print_info(df_chorzow)
df_ch_okresy = classify_weather_periods(df_chorzow)
calculate_dry_baseline(df_ch_okresy)


from business_metrics import (
    classify_weather_periods,
    calculate_dry_baseline,
    calculate_excess,
    summarize_rain_events,
)

# 1. Klasyfikacja godzin
df_chorzow = classify_weather_periods(df_chorzow)

# 2. Baseline suchej pogody
df_chorzow = calculate_dry_baseline(df_chorzow)

# 3. Nadwyżki i koszty
df_chorzow = calculate_excess(
    df_chorzow,
    price_per_kwh=1.0
)

# 4. Epizody deszczowe
episodes = summarize_rain_events(
    df_chorzow,
    price_per_kwh=1.0
)

print("Klasyfikacja godzin:")
print(df_chorzow["okres"].value_counts())

print("\nGodziny mokre z kompletnym oszacowaniem:")
wet = df_chorzow["okres"].eq("mokry")

print(
    df_chorzow.loc[wet, "nadwyzka_kwh"].notna().sum(),
    "/",
    wet.sum()
)

print("\n10 najdroższych epizodów:")
print(
    episodes[
        ["poczatek", "koniec", "godziny",
         "nadwyzka_m3", "nadwyzka_kwh", "koszt_zl"]
    ].head(10)
)

print("\nSzacowany koszt kompletnych epizodów:")
print(
    round(episodes["koszt_zl"].sum(), 2),
    "zł"
)


dry = df_chorzow["okres"].eq("suchy")
wet = df_chorzow["okres"].eq("mokry")

print("Średnie obserwowane wartości:")

print(
    df_chorzow.groupby("okres")[
        ["przepompowano_m3", "energia_kwh"]
    ].mean().round(2)
)

print("\nLiczba mokrych godzin bez baseline'u:")
print(
    df_chorzow.loc[
        wet,
        ["bazowe_m3", "bazowe_kwh"]
    ].isna().any(axis=1).sum()
)

print("\nBaseline w przykładowych mokrych godzinach:")
print(
    df_chorzow.loc[
        wet,
        [
            "przepompowano_m3",
            "bazowe_m3",
            "energia_kwh",
            "bazowe_kwh",
            "nadwyzka_m3",
            "nadwyzka_kwh"
        ]
    ].head(10)
)