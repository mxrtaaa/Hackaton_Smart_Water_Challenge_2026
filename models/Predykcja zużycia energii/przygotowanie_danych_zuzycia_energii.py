
import pandas as pd
from pathlib import Path
import sys

# Import funkcji z folderu src
src_path = Path(__file__).resolve().parents[2] / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from sciezka_projektu import get_project_root
from data_preparation import convert_meter_data, set_timezone


def przygotuj_dane_zuzycia_energii():

    PROJECT_ROOT = get_project_root()
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "Chorzów"

    # WCZYTANIE DANYCH

    df_ch_energia = pd.read_parquet(
        DATA_PATH / "energia.parquet"
    )

    df_ch_opady = pd.read_parquet(
        DATA_PATH / "suma_opadow.parquet"
    )

    # ENERGIA — godzinowe przyrosty licznika

    df_1h = convert_meter_data(
        df_ch_energia,
        "Zuzycie_energii_elektrycznej",
        "zuzycie_energii"
    ).to_frame()

    # OPADY — ujednolicenie czasu i agregacja godzinowa

    df_ch_opady = set_timezone(df_ch_opady)

    df_ch_opady["suma_opadu_mm"] = pd.to_numeric(
        df_ch_opady["suma_opadu_mm"]
        .astype(str)
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )

    opady_h = (
        df_ch_opady["suma_opadu_mm"]
        .groupby(level=0).sum(min_count=1)
        .resample("h").sum(min_count=1)
        .rename("suma_opadu_mm")
    )

    # ŁĄCZENIE ENERGII I OPADÓW

    df_lgbm = df_1h.join(opady_h, how="left")

    # CECHY KALENDARZOWE

    # Indeks pozostaje w stałym UTC+1.
    # Cechy godzinowe liczymy z polskiego czasu lokalnego.
    local_time = df_lgbm.index.tz_convert("Europe/Warsaw")

    df_lgbm["hour"] = local_time.hour
    df_lgbm["dayofweek"] = local_time.dayofweek
    df_lgbm["month"] = local_time.month
    df_lgbm["weekend"] = (
        local_time.dayofweek >= 5
    ).astype(int)

    # CECHY OPADOWE

    r = df_lgbm["suma_opadu_mm"]

    for k in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        df_lgbm[f"opad_{k}h"] = (
            r.shift(1).rolling(k).sum()
        )

    df_lgbm["opad_max3h"] = (
        r.shift(1).rolling(3).max()
    )

    last_rain = (
        df_lgbm.index.to_series()
        .where(r.shift(1) > 0.1)
        .ffill()
    )

    df_lgbm["h_od_deszczu"] = (
        (
            df_lgbm.index.to_series() - last_rain
        ).dt.total_seconds() / 3600
    ).clip(upper=240)

    # HISTORIA ZUŻYCIA ENERGII

    energia = df_lgbm["zuzycie_energii"]

    for k in [1, 3, 6, 24, 168]:
        df_lgbm[f"energia_lag_{k}h"] = energia.shift(k)

    df_lgbm["energia_mean_6h"] = (
        energia.shift(1).rolling(6).mean()
    )

    df_lgbm["energia_mean_24h"] = (
        energia.shift(1).rolling(24).mean()
    )

    # WYBÓR CECH

    features = [
        "hour", "dayofweek", "month", "weekend",

        "opad_1h", "opad_3h", "opad_6h",
        "opad_12h", "opad_24h", "opad_48h",
        "opad_72h", "opad_168h",

        "opad_max3h", "h_od_deszczu",

        "energia_lag_1h", "energia_lag_3h",
        "energia_lag_6h", "energia_lag_24h",
        "energia_lag_168h",

        "energia_mean_6h", "energia_mean_24h"
    ]

    X = df_lgbm[features].copy()
    y = df_lgbm["zuzycie_energii"].copy()

    # Usuwamy tylko godziny z brakującym pomiarem celu.
    # Luki pozostają zachowane podczas liczenia lagów.
    valid = y.notna()

    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    return X, y
