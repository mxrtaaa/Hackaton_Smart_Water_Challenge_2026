import pandas as pd
from pathlib import Path

from data_preparation import convert_meter_data, set_timezone
from business_metrics import run_business_analysis

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

wynik_chorzow = run_business_analysis(df_chorzow)
print(f"\n========== CHORZÓW ==========")
print(wynik_chorzow["summary"])
print(wynik_chorzow["episodes"].head(10))
print(wynik_chorzow["monthly"])

# eksport do plików

# Path("reports/business/chorzow").mkdir(parents=True, exist_ok=True)

# wynik["episodes"].to_csv("reports/business/chorzow/epizody.csv")
# wynik["monthly"].to_csv("reports/business/chorzow/miesiace.csv")

# RUDA ŚLĄSKA


# RUDA ŚLĄSKA

from business_metrics import classify_weather_periods

RUDA_PATH = Path("./data/processed/Ruda Śląska")

# Opady godzinowe Open-Meteo
df_rs_opady = pd.read_parquet(
    RUDA_PATH / "suma_opadow.parquet"
)

df_rs_opady = set_timezone(df_rs_opady)

df_rs_opady["suma_opadu_mm"] = pd.to_numeric(
    df_rs_opady["suma_opadu_mm"], errors="coerce"
)

df_rs_opady_h = (
    df_rs_opady["suma_opadu_mm"]
    .groupby(level=0).sum(min_count=1)
    .resample("h").sum(min_count=1)
    .rename("opad_mm")
)


for stacja in ["weteranow", "zmp"]:

    print(f"\n========== {stacja.upper()} ==========")

    scada = pd.read_parquet(
        RUDA_PATH / f"scada-{stacja}.parquet",
        columns=["DataCzas", "LicznikSciekow"]
    )

    # Czas normalizujemy PRZED sortowaniem.
    scada = set_timezone(scada, time_col="DataCzas")

    scada["LicznikSciekow"] = pd.to_numeric(
        scada["LicznikSciekow"].astype(str)
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )

    print(
        "Poprawne odczyty licznika:",
        scada["LicznikSciekow"].notna().sum()
    )
    print(
        "Różne wartości licznika:",
        scada["LicznikSciekow"].nunique()
    )

    # Narastający licznik w m3
    licznik_h = (
        scada["LicznikSciekow"]
        .dropna()
        .groupby(level=0).max()
        .sort_index()
        .resample("h").last()
    )

    przepompowano_h = (
        licznik_h.diff()
        .where(lambda x: x >= 0)
        .rename("przepompowano_m3")
    )

    print("\nRozkład godzinowych przyrostów:")
    print(
        przepompowano_h.quantile(
            [0.5, 0.90, 0.95, 0.99, 0.999]
        )
    )

    print("Godziny powyżej 100 m³:", (przepompowano_h > 100).sum())
    print("Godziny powyżej 1000 m³:", (przepompowano_h > 1000).sum())

    df_ruda = przepompowano_h.to_frame().join(
        df_rs_opady_h, how="left"
    )

    # Nie usuwamy godzin przed rollingiem.
    df_ruda = classify_weather_periods(df_ruda)

    print("\nPrzepływ według rodzaju pogody:")
    print(
        df_ruda.groupby("okres")["przepompowano_m3"]
        .agg(["count", "mean", "median"])
        .round(2)
    )

    print("\nBrakujące godzinowe przepływy:")
    print(df_ruda["przepompowano_m3"].isna().sum())

    print("\nNajwiększe przyrosty m3:")
    print(df_ruda["przepompowano_m3"].nlargest(5))



# ENERGIA RUDY ŚLĄSKIEJ — WETERANÓW I ZMP

wyniki_ruda = {}

for stacja in ["weteranow", "zmp"]:

    energia = pd.read_parquet(
        RUDA_PATH / f"energia-{stacja}.parquet",
        columns=["Data_czas", "Strefa", "Wartosc_kWh"]
    )

    # Niejednoznaczne jesienne godziny pomijamy.
    energia = set_timezone(
        energia,
        ambiguous="NaT"
    )

    energia = energia.loc[
        energia.index.notna()
    ].copy()

    energia["Wartosc_kWh"] = pd.to_numeric(
        energia["Wartosc_kWh"].astype(str)
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )

    # Sprawdzamy, czy każdej godzinie odpowiada
    # najwyżej jeden pomiar.
    if energia.index.has_duplicates:
        raise ValueError(
            f"{stacja}: zduplikowane daty energii. "
            "Sprawdź rekordy przed agregacją."
        )

    # Energia jest już zużyciem w interwale,
    # dlatego nie używamy .diff().
    energia_h = (
        energia["Wartosc_kWh"]
        .sort_index()
        .asfreq("h")
        .rename("energia_kwh")
    )

    # Opady z poprzedniego bloku analizy Rudy
    df = energia_h.to_frame().join(
        opady_h, how="left"
    )

    df = classify_weather_periods(df)

    # Tylko okresy suche z poprawnym pomiarem
    dry = df.loc[
        df["okres"].eq("suchy")
        & df["energia_kwh"].notna()
    ]

    # Typowe zużycie energii w miesiącu i godzinie
    stats = dry.groupby(
        ["miesiac", "godzina"]
    )["energia_kwh"].agg(["median", "count"])

    # Wymagamy co najmniej 5 suchych obserwacji
    stats.loc[
        stats["count"] < 5, "median"
    ] = float("nan")

    df = df.join(
        stats["median"].rename("bazowe_kwh"),
        on=["miesiac", "godzina"]
    )

    # Fallback: typowe zużycie o tej godzinie
    # ze wszystkich suchych okresów
    fallback = dry.groupby(
        "godzina"
    )["energia_kwh"].median()

    df["bazowe_kwh"] = (
        df["bazowe_kwh"]
        .fillna(df["godzina"].map(fallback))
    )

    wet = df["okres"].eq("mokry")

    df["nadwyzka_kwh"] = (
        df["energia_kwh"] - df["bazowe_kwh"]
    ).where(wet)

    # Cena: 1 zł/kWh
    df["bilans_zl"] = df["nadwyzka_kwh"]

    wyniki_ruda[stacja] = df

    print(f"\n===== {stacja.upper()} =====")

    print("\nŚrednie zużycie energii:")
    print(
        df.groupby("okres")["energia_kwh"]
        .agg(["count", "mean", "median"])
        .round(2)
    )

    print("\nMokre godziny z wyceną:")
    print(
        df.loc[wet, "nadwyzka_kwh"].count(),
        "/",
        wet.sum()
    )

    print("\nBilans energii względem suchego baseline'u:")
    print(
        round(df["nadwyzka_kwh"].sum(), 2),
        "kWh"
    )

    print("\nBilans przy 1 zł/kWh:")
    print(
        round(df["bilans_zl"].sum(), 2),
        "zł"
    )

