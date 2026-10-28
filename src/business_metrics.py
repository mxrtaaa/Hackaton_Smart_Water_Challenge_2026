import pandas as pd
import numpy as np


CENA_ENERGII = 1.0  # zł/kWh

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

    local_time = result.index.tz_convert("Europe/Warsaw")

    result["miesiac"] = local_time.month
    result["godzina"] = local_time.hour
    result["weekend"] = (local_time.dayofweek >= 5).astype(int)

    return result


def calculate_dry_baseline(df, min_samples=5, *, dry_only=True):
    """Mediany profilu kalendarzowego; domyślnie tylko z suchych godzin.

    dry_only=False jest używane przez calculate_calendar_baseline, aby
    porównanie bez pogody zachowało te same grupy i minimum próbek.
    """
    result = df.copy()

    required = {
        "miesiac", "godzina", "weekend",
        "przepompowano_m3", "energia_kwh"
    }
    if dry_only:
        required.add("okres")

    missing = required - set(result.columns)
    if missing:
        raise KeyError(f"Brakuje kolumn: {sorted(missing)}")

    valid = result[["przepompowano_m3", "energia_kwh"]].notna().all(axis=1)
    if dry_only:
        valid &= result["okres"].eq("suchy")
    reference = result.loc[valid].copy()

    if len(reference) < min_samples:
        raise ValueError("Za mało kompletnych godzin suchych." if dry_only
                         else "Za mało kompletnych godzin profilu kalendarzowego.")

    result["bazowe_m3"] = np.nan
    result["bazowe_kwh"] = np.nan

    result["baseline_zrodlo"] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )

    result["baseline_n"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )

    levels = [(
            "miesiac+godzina+weekend",
            ["miesiac", "godzina", "weekend"]
        ), (
            "miesiac+godzina",
            ["miesiac", "godzina"]
        ), (
            "godzina+weekend",
            ["godzina", "weekend"]
        ),(
            "godzina",
            ["godzina"]
        ),(
            "miesiac",
            ["miesiac"]
        )
    ]

    for label, keys in levels:
        stats = reference.groupby(keys).agg(
            candidate_m3=("przepompowano_m3", "median"),
            candidate_kwh=("energia_kwh", "median"),
            n=("przepompowano_m3", "size")
        )

        candidates = result[keys].join(stats, on=keys)

        # Wypełniamy wyłącznie jeszcze nieuzupełnione
        # baseline'y i tylko z odpowiednio dużej próbki.
        use = (
            result["bazowe_m3"].isna()
            & candidates["n"].ge(min_samples)
            & candidates[
                ["candidate_m3", "candidate_kwh"]
            ].notna().all(axis=1)
        )

        result.loc[use, "bazowe_m3"] = (
            candidates.loc[use, "candidate_m3"]
        )

        result.loc[use, "bazowe_kwh"] = (
            candidates.loc[use, "candidate_kwh"]
        )

        result.loc[use, "baseline_zrodlo"] = label

        result.loc[use, "baseline_n"] = (
            candidates.loc[use, "n"].astype("Int64")
        )

    return result


def calculate_calendar_baseline(df, min_samples=5):
    """Profil bez pogody z wszystkich kompletnych historycznych m³ i kWh.

    Nie wymaga opadów ani klasyfikacji okresów. Korzysta z tych samych median,
    grup kalendarzowych i kolejności fallback jak calculate_dry_baseline.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("Wymagany jest DatetimeIndex ze strefą czasową.")
    if df.index.has_duplicates:
        raise ValueError("Indeks zawiera duplikaty dat.")
    result = df.copy()
    local_time = result.index.tz_convert("Europe/Warsaw")
    result["miesiac"] = local_time.month
    result["godzina"] = local_time.hour
    result["weekend"] = (local_time.dayofweek >= 5).astype(int)
    return calculate_dry_baseline(result, min_samples, dry_only=False)



def calculate_excess(df, price_per_kwh=1.0, *, wet_only=True):
    """Różnice względem baseline'u, domyślnie w okresach mokrych.

    wet_only=False pozwala wycenić wskazane godziny anomalii Q97 niezależnie
    od opadów, z przekazanym baseline'em modelowym lub kalendarzowym.
    """

    result = df.copy()

    required = [
        "przepompowano_m3",
        "energia_kwh",
        "bazowe_m3",
        "bazowe_kwh"
    ]

    valid = result[required].notna().all(axis=1)
    if wet_only:
        valid &= result["okres"].eq("mokry")

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


def calculate_anomaly_cost(df, price_per_kwh=CENA_ENERGII):
    """Koszt netto wskazanych godzin jednej anomalii względem bazowych kWh.

    Wymaga kompletu danych dla każdej godziny. Zachowuje ujemne różnice
    godzinowe, a do zera obcina dopiero ich sumę, jak summarize_rain_events.
    df zawiera rzeczywiste m³/kWh oraz bazowe_m3 i bazowe_kwh danego wariantu.
    """
    detail = calculate_excess(df, price_per_kwh, wet_only=False)
    hours = len(detail)
    priced = int(detail["nadwyzka_kwh"].count())
    complete = hours > 0 and priced == hours
    excess = detail["nadwyzka_kwh"].sum() if complete else np.nan
    return {"godziny": hours, "godziny_wycenione": priced, "kompletny": complete,
            "nadwyzka_kwh": excess,
            "koszt_zl": max(0.0, excess) * price_per_kwh if complete else np.nan}


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

def run_business_analysis(
    df,
    price=1.0,
    dry_mm_48h=0.2,
    wet_mm_24h=1.0,
    min_samples=5
):
    """
    df: godzinowy DataFrame z kolumnami:
        przepompowano_m3, energia_kwh, opad_mm
    """

    # 1. Podział na okresy suche i mokre
    df = classify_weather_periods(
        df,
        dry_mm_48h=dry_mm_48h,
        wet_mm_24h=wet_mm_24h
    )

    # 2. Bazowy przepływ i zużycie energii
    df = calculate_dry_baseline(
        df,
        min_samples=min_samples
    )

    # 3. Nadwyżki
    df = calculate_excess(
        df,
        price_per_kwh=price
    )

    # 4. Ranking epizodów
    episodes = summarize_rain_events(
        df,
        price_per_kwh=price
    )

    # 5. Podsumowanie miesięczne
    wet = df.loc[df["okres"] == "mokry"].copy()

    wet["miesiac"] = (
        wet.index
        .tz_convert("Europe/Warsaw")
        .strftime("%Y-%m")
    )

    monthly = wet.groupby("miesiac").agg(
        godziny_mokre=("okres", "size"),
        godziny_wycenione=("nadwyzka_kwh", "count"),
        nadwyzka_m3=(
            "nadwyzka_m3",
            lambda x: x.sum(min_count=1)
        ),
        nadwyzka_kwh=(
            "nadwyzka_kwh",
            lambda x: x.sum(min_count=1)
        )
    )

    monthly["pokrycie"] = (
        monthly["godziny_wycenione"]
        / monthly["godziny_mokre"]
    )

    monthly["bilans_zl"] = (
        monthly["nadwyzka_kwh"] * price
    )

    # 6. Najważniejsze liczby
    summary = pd.Series({
        "godziny_mokre": len(wet),
        "godziny_wycenione": wet["nadwyzka_kwh"].count(),
        "pelne_epizody": int(episodes["kompletny"].sum()),
        "koszt_pelnych_epizodow_zl": (
            episodes["koszt_zl"].sum()
        )
    })

    return {
        "detail": df,
        "episodes": episodes,
        "monthly": monthly,
        "summary": summary
    }
