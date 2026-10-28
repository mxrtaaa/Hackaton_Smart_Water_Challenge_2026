from datetime import timedelta, timezone
from pathlib import Path
import sys
import numpy as np
import pandas as pd

src_path = Path(__file__).resolve().parents[2] / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from sciezka_projektu import get_project_root


def fix_tz_local(df, time_col="Data_czas"):
    res = df.copy()
    if time_col in res.columns:
        res[time_col] = pd.to_datetime(res[time_col])
        res = res.set_index(time_col)

    # Kluczowe: sortowanie i usunięcie zduplikowanych timestampów przed tz_localize
    res = res.sort_index()
    res = res[~res.index.duplicated(keep="first")]

    if res.index.tz is None:
        res.index = res.index.tz_localize(
            "Europe/Warsaw",
            ambiguous="NaT",
            nonexistent="shift_forward",
        )

    res = res[res.index.notna()]
    res.index = res.index.tz_convert(timezone(timedelta(hours=1)))
    return res


def przygotuj_dane_przepompowanie():
    PROJECT_ROOT = get_project_root()
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "Rybnik"

    df_r_pz1 = pd.read_parquet(DATA_PATH / "pz1-2024.parquet")
    df_r_pnb1 = pd.read_parquet(DATA_PATH / "pnb1-2024.parquet")
    df_r_opady = pd.read_parquet(DATA_PATH / "suma_opadow.parquet")

    # 1. Strefa czasowa i czyszczenie
    wyniki = []
    for df in [df_r_pz1,df_r_pnb1]:
        df = fix_tz_local(df)
        df = df.rename(columns={"Wartosc": "ilosc"})

        # ODCINAMY AWARIĘ LICZNIKA (bierzemy dane dopiero od poprawnego momentu):
        df = df.loc["2024-03-19 13:00:00":].copy()


        df["ilosc"] = pd.to_numeric(
            df["ilosc"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        df = df[df["ilosc"].notna()].sort_index()

        # 2. Strefa czasowa i agregacja opadów
        df_r_opady = fix_tz_local(df_r_opady)
        df_r_opady = df_r_opady.groupby(level=0).agg({"suma_opadu_mm": "sum"})

        # 3. Godzinowy przyrost przepływu z licznika narastającego
        hourly = df["ilosc"].resample("1h").last()
        df_1h = pd.DataFrame({"przepompowano": hourly.diff()})
        df_1h.loc[df_1h["przepompowano"] < 0, "przepompowano"] = np.nan

        # 4. Złączenie i cechy
        df_lgbm = df_1h.join(df_r_opady[["suma_opadu_mm"]], how="left")
        df_lgbm["suma_opadu_mm"] = df_lgbm["suma_opadu_mm"].fillna(0.0)

        # Kalendarz
        df_lgbm["hour"] = df_lgbm.index.hour
        df_lgbm["dayofweek"] = df_lgbm.index.day_of_week
        df_lgbm["month"] = df_lgbm.index.month
        df_lgbm["weekend"] = (df_lgbm.index.day_of_week >= 5).astype(int)

        # Cykliczność
        df_lgbm["hour_sin"] = np.sin(2 * np.pi * df_lgbm.index.hour / 24)
        df_lgbm["hour_cos"] = np.cos(2 * np.pi * df_lgbm.index.hour / 24)
        df_lgbm["month_sin"] = np.sin(2 * np.pi * (df_lgbm.index.month - 1) / 12)
        df_lgbm["month_cos"] = np.cos(2 * np.pi * (df_lgbm.index.month - 1) / 12)

        # Cechy opadów
        r = df_lgbm["suma_opadu_mm"]
        df_lgbm["opad_0h"] = r
        df_lgbm["opad_lag_1h"] = r.shift(1)
        df_lgbm["opad_max6h"] = r.rolling(6, min_periods=1).max()
        df_lgbm["opad_max3h"] = r.rolling(3, min_periods=1).max()

        for k in [2, 3, 6, 12, 24, 48]:
            df_lgbm[f"opad_{k}h"] = r.rolling(k, min_periods=1).sum()
            df_lgbm[f"opad_lag_{k}h"] = r.shift(k)

        last_rain = df_lgbm.index.to_series().where(r > 0.1).ffill()
        df_lgbm["h_od_deszczu"] = (
            (df_lgbm.index.to_series() - last_rain).dt.total_seconds() / 3600
        ).fillna(240).clip(upper=240)

        features = [
            "hour", "dayofweek", "month", "weekend",
            "month_sin", "month_cos",
            "opad_0h", "opad_2h", "opad_3h", "opad_6h",
            "opad_12h", "opad_24h", "opad_48h",
            "opad_max6h",
            "opad_lag_1h", "opad_lag_2h",
            "opad_lag_3h", "opad_lag_6h",
            "opad_lag_12h", "opad_lag_24h",
        ]

        # Usunięcie wierszy z brakami
        mask = df_lgbm["przepompowano"].notna() & df_lgbm[features].notna().all(axis=1)
        X = df_lgbm.loc[mask, features].copy()
        y = df_lgbm.loc[mask, "przepompowano"].copy()
        wyniki.append((X,y))
    X_pz1 = wyniki[0][0]
    y_pz1 = wyniki[0][1]
    X_pnb1 = wyniki[1][0]
    y_pnb1 = wyniki[1][1]
    return X_pz1, y_pz1, X_pnb1, y_pnb1