import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import timedelta, timezone

src_path = Path().resolve().parents[1] / 'src'
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from scada_prepare_data import load_scada_weterani
from sciezka_projektu import get_project_root
from data_preparation import set_timezone

def fix_tz_local(df, time_col="Data_czas"):
    df = df.copy()
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

    df = df.sort_index()

    if df.index.tz is None:
        # Tu omijamy błąd 31 marca i 27 października bez ruszania data_preparation.py:
        df.index = df.index.tz_localize(
            "Europe/Warsaw", ambiguous="NaT", nonexistent="shift_forward"
        )

    df = df[df.index.notna()]
    df.index = df.index.tz_convert(timezone(timedelta(hours=1)))
    return df

def przygotuj_dane():
    PROJECT_ROOT = get_project_root()

    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "Ruda Śląska"
    df_rs_opady = pd.read_parquet(DATA_PATH / "suma_opadow.parquet")
    df_weteranow = load_scada_weterani()

    df_weteranow = fix_tz_local(df_weteranow)
    df_rs_opady = fix_tz_local(df_rs_opady)

    df_rs_opady = df_rs_opady.groupby(level=0).agg({"suma_opadu_mm": "sum"})

    df_lgbm = df_weteranow.join(df_rs_opady[["suma_opadu_mm"]], how="left")

    # Dajemy modelowi kontekst kalendarzowy, ponieważ może zależeć od pory dnia i dnia tygodnia
    df_lgbm["hour"] = df_lgbm.index.hour
    df_lgbm["dayofweek"] = df_lgbm.index.day_of_week
    df_lgbm["month"] = df_lgbm.index.month
    df_lgbm["weekend"] = (df_lgbm.index.day_of_week >= 5).astype(int)

    #Żeby mieć cykliczność opadów
    df_lgbm["hour_sin"] = np.sin(2 * np.pi * df_lgbm.index.hour / 24)
    df_lgbm["hour_cos"] = np.cos(2 * np.pi * df_lgbm.index.hour / 24)

    # Cykliczność roku
    df_lgbm["month_sin"] = np.sin(2 * np.pi * (df_lgbm.index.month - 1) / 12)
    df_lgbm["month_cos"] = np.cos(2 * np.pi * (df_lgbm.index.month - 1) / 12)

    r = df_lgbm["suma_opadu_mm"]

    df_lgbm["opad_lag_1h"] = r.shift(1)
    df_lgbm["opad_max6h"] = r.rolling(6).max()


    df_lgbm["opad_0h"] = r  # bieżąca godzina
    #suma opadów z 2/3/6h itd
    for k in [2, 3, 6, 12, 24, 48]:                                             #JAK DALEKO W TYŁ MODEL MA PATRZEĆ NA OPADY (godziny)
        df_lgbm[f"opad_{k}h"] = r.rolling(k).sum()

    df_lgbm["opad_max3h"] = r.rolling(3).max()  # intensywność szczytowa

    last_rain = df_lgbm.index.to_series().where(r > 0.1).ffill()
    df_lgbm["h_od_deszczu"] = ((df_lgbm.index.to_series() - last_rain)
                               .dt.total_seconds() / 3600).clip(upper=240)

    #opóźnienia opadów - czy może przepompownia reaguje na opady z opóźnieniem
    for k in [2, 3, 6, 12, 24, 48]:
        df_lgbm[f"opad_lag_{k}h"] = r.shift(k)


    features = [
        # Kalendarzowe
        "hour", "dayofweek", "month", "weekend",
        "month_sin", "month_cos",

        # Opady
        "opad_0h", "opad_2h", "opad_3h", "opad_6h",
        "opad_12h", "opad_24h", "opad_48h",
        "opad_max6h",

        # Opóźnienia opadów
        "opad_lag_1h", "opad_lag_2h",
        "opad_lag_3h", "opad_lag_6h",
        "opad_lag_12h", "opad_lag_24h"
    ]


    X = df_lgbm[features].copy()
    y = df_lgbm["przepompowano"].copy()
    # Usuwamy tylko wiersze z brakującą wartością docelową
    mask = y.notna()

    X = X.loc[mask].copy()
    y = y.loc[mask].copy()
    return X,y
