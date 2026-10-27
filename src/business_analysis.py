"""Analiza biznesowa Chorzowa i Rudy Śląskiej.

Uruchom bez argumentów w VS Code (Run Python File).
Dane wynikowe Rudy są też zapisywane godzinowo do użycia przez modele.
Nie zmienia współdzielonych funkcji z data_preparation.py.
"""

from datetime import timedelta, timezone

import pandas as pd

from business_metrics import (
    classify_weather_periods,
    run_business_analysis,
)
from data_preparation import convert_meter_data, set_timezone
from sciezka_projektu import get_project_root


ROOT = get_project_root()
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CENA_KWH = 1.0

# Ustalony roboczo przez zespół próg dla Weteranów.
# Nie jest potwierdzoną maksymalną wydajnością przepompowni.
MAX_M3_H_WETERANOW = 100.0
STALA_STREFA = timezone(timedelta(hours=1))


# ============================================================================
# CHORZÓW — pozostawiamy sprawdzone przetwarzanie
# ============================================================================
print("\n========== CHORZÓW ==========")

scieki = pd.read_parquet(DATA / "Chorzów" / "scieki.parquet")
energia = pd.read_parquet(DATA / "Chorzów" / "energia.parquet")
opady = pd.read_parquet(DATA / "Chorzów" / "suma_opadow.parquet")

scieki_h = convert_meter_data(
    scieki, "Ilosc_przepompowanych_sciekow", "przepompowano_m3"
)
energia_h = convert_meter_data(
    energia, "Zuzycie_energii_elektrycznej", "energia_kwh"
)

opady = set_timezone(opady)
opady["suma_opadu_mm"] = pd.to_numeric(opady["suma_opadu_mm"], errors="coerce")
opady_h = (
    opady["suma_opadu_mm"]
    .groupby(level=0).sum(min_count=1)
    .resample("h").sum(min_count=1)
    .rename("opad_mm")
)

df_chorzow = pd.concat([scieki_h, energia_h], axis=1).sort_index()
df_chorzow = df_chorzow.join(opady_h, how="left")
wynik_chorzow = run_business_analysis(df_chorzow, price=CENA_KWH)

print(wynik_chorzow["summary"])
print("\nNajdroższe epizody:")
print(wynik_chorzow["episodes"].head(5))

folder = REPORTS / "chorzow"
folder.mkdir(parents=True, exist_ok=True)
wynik_chorzow["detail"].to_parquet(folder / "godziny.parquet")
wynik_chorzow["episodes"].to_csv(folder / "epizody.csv")
wynik_chorzow["monthly"].to_csv(folder / "miesiace.csv")


# ============================================================================
# RUDA ŚLĄSKA — wspólne opady, potem każda przepompownia osobno
# ============================================================================
RUDA = DATA / "Ruda Śląska"

opady = pd.read_parquet(RUDA / "suma_opadow.parquet")
opady = set_timezone(opady)
opady["suma_opadu_mm"] = pd.to_numeric(opady["suma_opadu_mm"], errors="coerce")
opady_h = (
    opady["suma_opadu_mm"]
    .groupby(level=0).sum(min_count=1)
    .resample("h").sum(min_count=1)
    .rename("opad_mm")
)

for stacja in ("weteranow", "zmp"):
    print(f"\n========== RUDA: {stacja.upper()} ==========")

    # SCADA: surowy, narastający licznik m³. Nie wywołujemy
    # load_scada_weterani(), bo ta funkcja już oblicza przyrosty.
    scada = pd.read_parquet(
        RUDA / f"scada-{stacja}.parquet",
        columns=["DataCzas", "LicznikSciekow"],
    )

    # Daty ze SCADA są polskim czasem lokalnym bez strefy.
    # Podczas zmian czasu nie próbujemy zgadywać godzin 02:00.
    czas = pd.DatetimeIndex(pd.to_datetime(scada["DataCzas"], errors="coerce"))
    if czas.tz is None:
        czas = czas.tz_localize(
            "Europe/Warsaw", ambiguous="NaT", nonexistent="NaT"
        )
    czas = czas.tz_convert(STALA_STREFA)

    print("Odrzucone niejednoznaczne/nieistniejące daty SCADA:", int(czas.isna().sum()))
    scada.index = czas
    scada = scada.loc[scada.index.notna()].copy()
    scada.index.name = "Data_czas"

    scada["LicznikSciekow"] = pd.to_numeric(
        scada["LicznikSciekow"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    # Najpierw godziny, dopiero później różnica licznika.
    # Brak pomiaru w godzinie pozostaje NaN (bez ffill/interpolacji).
    licznik_h = (
        scada["LicznikSciekow"]
        .dropna()
        .groupby(level=0).max()
        .sort_index()
        .resample("h").last()
    )
    przyrost = licznik_h.diff()
    bledne = przyrost < 0
    if stacja == "weteranow":
        bledne = bledne | (przyrost > MAX_M3_H_WETERANOW)

    przepompowano_h = przyrost.mask(bledne).rename("przepompowano_m3")
    print("Odrzucone godzinowe skoki/cofnięcia licznika:", int(bledne.sum()))
    print("Kwantyle przyrostów po filtracji:")
    print(przepompowano_h.quantile([0.5, 0.9, 0.99, 0.999]))

    # ENERGIA: według dotychczasowych danych — godzinowe kWh,
    # a nie licznik narastający. Dlatego NIE wywołujemy diff().
    energia = pd.read_parquet(
        RUDA / f"energia-{stacja}.parquet",
        columns=["Data_czas", "Strefa", "Wartosc_kWh"],
    )
    czas = pd.DatetimeIndex(pd.to_datetime(energia["Data_czas"], errors="coerce"))
    if czas.tz is None:
        czas = czas.tz_localize(
            "Europe/Warsaw", ambiguous="NaT", nonexistent="NaT"
        )
    czas = czas.tz_convert(STALA_STREFA)

    print("Odrzucone niejednoznaczne/nieistniejące daty energii:", int(czas.isna().sum()))
    energia.index = czas
    energia = energia.loc[energia.index.notna()].copy()
    energia.index.name = "Data_czas"

    energia["Wartosc_kWh"] = pd.to_numeric(
        energia["Wartosc_kWh"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    energia.loc[energia["Wartosc_kWh"] < 0, "Wartosc_kWh"] = float("nan")

    # Przy kilku odczytach w tej samej godzinie wybieramy pierwszy
    # chronologicznie. Dla identycznych znaczników czasu zachowujemy
    # kolejność z pliku źródłowego. Nie sumujemy godzinowych kWh.
    energia = energia.sort_index(kind="stable")
    wielokrotne_godziny = (
        energia["Wartosc_kWh"].resample("h").size() > 1
    )
    print("Godziny z wieloma odczytami energii (wybrano pierwszy):", int(wielokrotne_godziny.sum()))

    energia_h = (
        energia["Wartosc_kWh"]
        .resample("h").first()
        .rename("energia_kwh")
    )

    # Wspólne, godzinowe wejście do analizy i dalszych prac zespołu.
    df_ruda = pd.concat([przepompowano_h, energia_h], axis=1).sort_index()
    df_ruda = df_ruda.join(opady_h, how="left").asfreq("h")

    if stacja == "zmp":
        # Licznik ZMP praktycznie nie zmienia wartości — nie udajemy,
        # że zerowy przyrost oznacza brak ścieków. Zostawiamy go do kontroli.
        df_ruda["przepompowano_m3_scada"] = df_ruda["przepompowano_m3"]
        df_ruda["przepompowano_m3"] = float("nan")

    # Gotowe dane dla modeli: oryginalnych pomiarów nie interpolujemy.
    df_ruda.to_parquet(RUDA / f"godzinowe-{stacja}.parquet")
    print("Zapisano godzinowe dane:", RUDA / f"godzinowe-{stacja}.parquet")
    print("Braki w danych:")
    print(df_ruda[["przepompowano_m3", "energia_kwh", "opad_mm"]].isna().sum())

    folder = REPORTS / f"ruda-{stacja}"
    folder.mkdir(parents=True, exist_ok=True)

    if stacja == "weteranow":
        # Pełna analiza m³ + kWh, jeśli istnieją suche pomiary obu liczników.
        wynik = run_business_analysis(df_ruda, price=CENA_KWH)
        print("\nPodsumowanie biznesowe:")
        print(wynik["summary"])
        print("\nNajdroższe epizody:")
        print(wynik["episodes"].head(5))
        wynik["detail"].to_parquet(folder / "godziny.parquet")
        wynik["episodes"].to_csv(folder / "epizody.csv")
        wynik["monthly"].to_csv(folder / "miesiace.csv")

    else:
        # ZMP: tylko energia względem suchej pogody (brak wiarygodnych m³).
        df = classify_weather_periods(df_ruda)
        dry = df.loc[df["okres"].eq("suchy") & df["energia_kwh"].notna()]
        stats = dry.groupby(["miesiac", "godzina"])["energia_kwh"].agg(["median", "count"])
        stats.loc[stats["count"] < 5, "median"] = float("nan")
        df = df.join(stats["median"].rename("bazowe_kwh"), on=["miesiac", "godzina"])
        fallback = dry.groupby("godzina")["energia_kwh"].median()
        df["bazowe_kwh"] = df["bazowe_kwh"].fillna(df["godzina"].map(fallback))

        wet = df["okres"].eq("mokry")
        df["nadwyzka_kwh"] = (df["energia_kwh"] - df["bazowe_kwh"]).where(wet)
        df["bilans_zl"] = df["nadwyzka_kwh"] * CENA_KWH

        print("\nMokre godziny z wyceną:", int(df.loc[wet, "nadwyzka_kwh"].count()), "/", int(wet.sum()))
        print("Bilans kosztu energii w mokrych godzinach [zł]:", round(df["bilans_zl"].sum(), 2))
        df.to_parquet(folder / "godziny.parquet")
