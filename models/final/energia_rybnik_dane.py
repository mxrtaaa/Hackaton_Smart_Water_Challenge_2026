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

    # Jeśli czas jest w kolumnie, ustaw jako indeks
    if time_col in res.columns:
        res[time_col] = pd.to_datetime(res[time_col])
        res = res.set_index(time_col)
    elif not isinstance(res.index, pd.DatetimeIndex):
        possible_cols = [c for c in res.columns if "data" in c.lower() or "time" in c.lower()]
        if possible_cols:
            res[possible_cols[0]] = pd.to_datetime(res[possible_cols[0]])
            res = res.set_index(possible_cols[0])
        else:
            raise TypeError("Brak indeksu DatetimeIndex i nie znaleziono kolumny z czasem.")

    res = res.sort_index()
    res = res[~res.index.duplicated(keep="first")]

    if res.index.tz is None:
        res.index = res.index.tz_localize(
            "Europe/Warsaw",
            ambiguous=True,
            nonexistent="shift_forward"  # Rozwiązuje błąd luki 02:00:00 z 31.03.2024
        )

    res = res[res.index.notna()]
    target_tz = timezone(timedelta(hours=1))
    res.index = res.index.tz_convert(target_tz)

    return res


def przygotuj_dane_energia():
    PROJECT_ROOT = get_project_root()
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final"

    # WCZYTANIE DANYCH
    df_ch_energia_pz1 = pd.read_parquet(DATA_PATH / "godzinowe-pz1-final.parquet")
    df_ch_energia_pnb1 = pd.read_parquet(DATA_PATH / "godzinowe-pnb1-final.parquet")
    
    wyniki = []
    for df in [df_ch_energia_pz1, df_ch_energia_pnb1]: 
        # ENERGIA
        '''df_energia = fix_tz_local(df)

        df_energia["Wartosc_kwh"] = pd.to_numeric(
            df_energia["Wartosc_kwh"]
            .astype(str)
            .str.replace(",", ".", regex=False),
            errors="coerce"
        )

        df_1h = (
            df_energia["Wartosc_kWh_kvar"]
            .resample("1h")
            .sum()
            .to_frame(name="zuzycie_energii")
        )

        # OPADY
        df_opady = fix_tz_local(df_ch_opady)

        df_opady["suma_opadu_mm"] = pd.to_numeric(
            df_opady["suma_opadu_mm"]
            .astype(str)
            .str.replace(",", ".", regex=False),
            errors="coerce"
        )

        opady_h = (
            df_opady["suma_opadu_mm"]
            .groupby(level=0).sum(min_count=1)
            .resample("1h").sum(min_count=1)
            .rename("suma_opadu_mm")
        )'''

        # ŁĄCZENIE
        df_lgbm = df.copy()
        df_lgbm["opad_mm"] = df_lgbm["opad_mm"].fillna(0.0)

        # CECHY KALENDARZOWE
        df_lgbm["hour"] = df_lgbm.index.hour
        df_lgbm["dayofweek"] = df_lgbm.index.day_of_week
        df_lgbm["month"] = df_lgbm.index.month
        df_lgbm["weekend"] = (df_lgbm.index.day_of_week >= 5).astype(int)

        # CECHY CYKLICZNE (naprawione mnożenie 2 * np.pi)
        df_lgbm["hour_sin"] = np.sin(2 * np.pi * df_lgbm.index.hour / 24)
        df_lgbm["hour_cos"] = np.cos(2 * np.pi * df_lgbm.index.hour / 24)
        df_lgbm["month_sin"] = np.sin(2 * np.pi * (df_lgbm.index.month - 1) / 12)
        df_lgbm["month_cos"] = np.cos(2 * np.pi * (df_lgbm.index.month - 1) / 12)

        # CECHY OPADOWE
        r = df_lgbm["opad_mm"]

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

        energia = df_lgbm["energia_kwh"]

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

            "energia_mean_6h", "energia_mean_24h",
            "przepompowano_m3"
        ]

        X = df_lgbm[features].copy()
        y = df_lgbm["energia_kwh"].copy()

        # Usuwamy tylko godziny z brakującym pomiarem celu.
        # Luki pozostają zachowane podczas liczenia lagów.
        valid = y.notna()

        X = X.loc[valid].copy()
        y = y.loc[valid].copy()

        wyniki.append((X, y))

    X_pz1 = wyniki[0][0]
    y_pz1 = wyniki[0][1]
    X_pnb1 = wyniki[1][0]
    y_pnb1 = wyniki[1][1]

    return X_pz1, y_pz1, X_pnb1, y_pnb1