import pandas as pd
import numpy as np
from pathlib import Path
import sys

src_path = Path(__file__).resolve().parents[2] / 'src'
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from sciezka_projektu import get_project_root
def przygotuj_dane_zuzycia_energii():
    PROJECT_ROOT = get_project_root()

    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "Chorzów"

    df_ch_energia = pd.read_parquet(DATA_PATH / 'energia.parquet')
    df_ch_opady = pd.read_parquet(DATA_PATH / 'suma_opadow.parquet')

    df = df_ch_energia.copy()

    df["Sposób_pomiaru"] = df["Sposób_pomiaru"].astype(str).str.strip()
    df = df[df["Sposób_pomiaru"] == "Wartość z pomiaru"].copy()

    if "Data_czas" in df.columns:
        df["Data_czas"] = pd.to_datetime(df["Data_czas"])
        df = df.set_index("Data_czas")

    df = df.rename(columns={"Zuzycie_energii_elektrycznej": "energia"})

    # Konwersja na wartości liczbowe
    df["energia"] = pd.to_numeric(
        df["energia"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

    df = df[df["energia"].notna()]
    df = df.sort_index()

    #print(df[["energia"]].head(10))

    # Usuwamy zduplikowane daty, zachowując największy odczyt licznika energii
    df = df.groupby(level=0).agg({"energia": "max"})
    df = df.sort_index()

    df.head()

    # Godzinowe zużycie energii z licznika narastającego
    hourly = df["energia"].resample("1h").last()

    df_1h = pd.DataFrame({
        "zuzycie_energii": hourly.diff()
    })

    # Reset licznika lub nieprawidłowy odczyt
    df_1h.loc[df_1h["zuzycie_energii"] < 0, "zuzycie_energii"] = np.nan

    df_lgbm = df_1h.join(
        df_ch_opady[["suma_opadu_mm"]],
        how="left"
    )

    #print(df_lgbm.head(10))

    # Kontekst kalendarzowy dla prognozowania zużycia energii
    df_lgbm["hour"] = df_lgbm.index.hour
    df_lgbm["dayofweek"] = df_lgbm.index.day_of_week
    df_lgbm["month"] = df_lgbm.index.month
    df_lgbm["weekend"] = (df_lgbm["dayofweek"] >= 5).astype(int)

    #print(df_lgbm.head(10))

    r = df_lgbm["suma_opadu_mm"]

    # Opady z poprzednich godzin
    for k in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        df_lgbm[f"opad_{k}h"] = r.shift(1).rolling(k).sum()

    df_lgbm["opad_max3h"] = r.shift(1).rolling(3).max()

    last_rain = df_lgbm.index.to_series().where(r.shift(1) > 0.1).ffill()

    df_lgbm["h_od_deszczu"] = (
            (df_lgbm.index.to_series() - last_rain)
            .dt.total_seconds() / 3600
    ).clip(upper=240)

    #print(df_lgbm.head())

    energia = df_lgbm["zuzycie_energii"]

    for k in [1, 3, 6, 24, 168]:
        df_lgbm[f"energia_lag_{k}h"] = energia.shift(k)

    df_lgbm["energia_mean_6h"] = energia.shift(1).rolling(6).mean()
    df_lgbm["energia_mean_24h"] = energia.shift(1).rolling(24).mean()

    features = [
        "hour", "dayofweek", "month", "weekend",
        "opad_1h", "opad_3h", "opad_6h", "opad_12h",
        "opad_24h", "opad_48h", "opad_72h", "opad_168h",
        "opad_max3h", "h_od_deszczu",
        "energia_lag_1h", "energia_lag_3h", "energia_lag_6h",
        "energia_lag_24h", "energia_lag_168h",
        "energia_mean_6h", "energia_mean_24h"
    ]

    X = df_lgbm[features].copy()
    y = df_lgbm["zuzycie_energii"].copy()

    #print("Liczba cech:", X.shape[1])
    #print("Liczba obserwacji:", len(X))

    #print("\nBraki w cechach:", X.isna().sum())
    #print("\nBraki w y:", y.isna().sum())

    # Usuwamy tylko wiersze z brakującą wartością docelową
    valid_idx = y.notna()

    X = X.loc[valid_idx].copy()
    y = y.loc[valid_idx].copy()

    return X,y

