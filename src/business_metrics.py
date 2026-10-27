import pandas as pd
import numpy as np


CENA_ENERGII = 1.0  # zł/kWh

def print_info(df):
    print(df.head(10))
    print("\nBraki danych:")
    print(df.isna().sum())
    print("\nStrefa czasowa:", df.index.tz)

    df["koszt_energii_zl"] = (
        df["energia_kwh"] * CENA_ENERGII
    )

    print(
        "Łączny koszt energii:",
        round(df["koszt_energii_zl"].sum(), 2),
        "zł"
    )

def classify_weather_periods(
    df,
    dry_mm_48h=0.2,
    wet_mm_24h=1.0
):
    """Klasyfikacja godzin według historii opadów."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Wymagany jest indeks DatetimeIndex.")

    if df.index.tz is None:
        raise ValueError("Indeks musi mieć przypisaną strefę czasową.")

    if df.index.has_duplicates:
        raise ValueError("Indeks zawiera duplikaty dat.")

    # Zachowujemy regularną siatkę godzin, także z brakami.
    result = df.sort_index().asfreq("h").copy()

    if result["opad_mm"].lt(0).any():
        raise ValueError("Opady zawierają ujemne wartości.")

    result["opad_24h"] = (
        result["opad_mm"]
        .rolling(24, min_periods=24)
        .sum()
    )

    result["opad_48h"] = (
        result["opad_mm"]
        .rolling(48, min_periods=48)
        .sum()
    )

    known = result[["opad_24h", "opad_48h"]].notna().all(axis=1)

    result["okres"] = "nieokreslony"
    result.loc[known, "okres"] = "przejsciowy"

    result.loc[
        known & (result["opad_48h"] < dry_mm_48h),
        "okres"
    ] = "suchy"

    result.loc[
        known & (result["opad_24h"] >= wet_mm_24h),
        "okres"
    ] = "mokry"

    # Do profilu dobowego używamy czasu lokalnego.
    local_time = result.index.tz_convert("Europe/Warsaw")

    result["miesiac"] = local_time.month
    result["godzina"] = local_time.hour
    result["weekend"] = (local_time.dayofweek >= 5).astype(int)

    return result


def calculate_dry_baseline(df, min_samples=5):
    """Mediana przepływu i energii w podobnych suchych godzinach."""

    result = df.copy()
    features = ["miesiac", "godzina", "weekend"]

    dry = result.loc[
        result["okres"].eq("suchy")
        & result[
            ["przepompowano_m3", "energia_kwh"]
        ].notna().all(axis=1)
    ]

    if dry.empty:
        raise ValueError("Brak poprawnych pomiarów z okresów suchych.")

    baseline = dry.groupby(features).agg(
        bazowe_m3=("przepompowano_m3", "median"),
        bazowe_kwh=("energia_kwh", "median"),
        n_suchych=("energia_kwh", "size")
    )

    # Nie wyznaczamy baseline'u z bardzo małej próbki.
    insufficient = baseline["n_suchych"] < min_samples

    baseline.loc[
        insufficient,
        ["bazowe_m3", "bazowe_kwh"]
    ] = np.nan

    result = result.join(baseline, on=features)

    return result


def calculate_excess(df, price_per_kwh=1.0):
    """Różnice względem baseline'u w okresach mokrych."""

    result = df.copy()

    required = [
        "przepompowano_m3",
        "energia_kwh",
        "bazowe_m3",
        "bazowe_kwh"
    ]

    valid = (
        result["okres"].eq("mokry")
        & result[required].notna().all(axis=1)
    )

    # Zachowujemy również ujemne różnice.
    # Nie obcinamy każdej godziny do zera.
    result["nadwyzka_m3"] = (
        result["przepompowano_m3"] - result["bazowe_m3"]
    ).where(valid)

    result["nadwyzka_kwh"] = (
        result["energia_kwh"] - result["bazowe_kwh"]
    ).where(valid)

    result["koszt_nadwyzki_zl"] = (
        result["nadwyzka_kwh"] * price_per_kwh
    )

    return result


def summarize_rain_events(df, price_per_kwh=1.0):
    """Łączy sąsiednie mokre godziny w epizody."""

    wet_mask = df["okres"].eq("mokry")

    # Nowy epizod zaczyna się po godzinie niemokrej.
    starts = wet_mask & ~wet_mask.shift(fill_value=False)

    wet = df.loc[wet_mask].copy()

    wet["epizod_id"] = (
        starts.cumsum().loc[wet_mask].astype(int)
    )

    wet["czas"] = wet.index

    grouped = wet.groupby("epizod_id")

    episodes = grouped.agg(
        poczatek=("czas", "min"),
        koniec=("czas", "max"),
        godziny=("czas", "size"),
        godziny_z_pomiarami=("nadwyzka_kwh", "count")
    )

    totals = grouped[
        ["nadwyzka_m3", "nadwyzka_kwh"]
    ].sum(min_count=1)

    episodes = episodes.join(totals)

    # Nie podajemy pełnego kosztu epizodu, jeśli brakuje
    # choć jednej godziny potrzebnej do kalkulacji.
    episodes["kompletny"] = (
        episodes["godziny"] == episodes["godziny_z_pomiarami"]
    )

    episodes.loc[
        ~episodes["kompletny"],
        ["nadwyzka_m3", "nadwyzka_kwh"]
    ] = np.nan

    # Do rankingu bierzemy tylko dodatnią nadwyżkę netto
    # na poziomie całego epizodu.
    episodes["koszt_zl"] = (
        episodes["nadwyzka_kwh"].clip(lower=0)
        * price_per_kwh
    )

    return episodes.sort_values(
        "koszt_zl", ascending=False
    )