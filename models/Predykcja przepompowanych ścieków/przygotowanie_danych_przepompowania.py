import pandas as pd
import numpy as np
import sys
from pathlib import Path

src_path = Path(__file__).resolve().parents[2] / 'src'
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from sciezka_projektu import get_project_root


def przygotuj_dane_przepompowania_z_opadami():
    PROJECT_ROOT = get_project_root()

    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "Chorzów"

    df_ch_scieki = pd.read_parquet(DATA_PATH / 'scieki.parquet')
    df_ch_opady = pd.read_parquet(DATA_PATH / 'suma_opadow.parquet')


    df = df_ch_scieki.copy()

    df["Sposób_pomiaru"] = df["Sposób_pomiaru"].astype(str).str.strip()
    df = df[df["Sposób_pomiaru"] == "Wartość z pomiaru"].copy()

    if "Data_czas" in df.columns:
        df["Data_czas"] = pd.to_datetime(df["Data_czas"])
        df = df.set_index("Data_czas")

    df = df.rename(columns={"Ilosc_przepompowanych_sciekow": "ilosc"})

    # konwersja na liczbowe
    df["ilosc"] = pd.to_numeric(
        df["ilosc"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

    # usuwanie zer i śmieci
    df = df[df["ilosc"].notna()]
    #print(df[["ilosc"]].head(10))



    # Mamy zduplikowane daty (najprawdopodobniej wynikają ze zmiany czasu na zimowy/letni), więc zachowamy te największe odczyty dla każdej daty
    df = df.groupby(level=0).agg({"ilosc": "max"})
    df = df.sort_index()

    df.head()


    df_ch_opady["Data_czas"] = pd.to_datetime(df_ch_opady["Data_czas"])
    df_ch_opady = df_ch_opady.set_index("Data_czas").sort_index()

    #print(df_ch_opady.head())

    hourly = df["ilosc"].resample("1h").last()
    df_1h = pd.DataFrame({"przepompowano": hourly.diff()})

    # reset licznika
    df_1h.loc[df_1h["przepompowano"] < 0, "przepompowano"] = np.nan

    df_lgbm = df_1h.join(df_ch_opady[["suma_opadu_mm"]], how="left")
    df_lgbm["suma_opadu_mm"] = df_lgbm["suma_opadu_mm"].fillna(0)

    #print(df_lgbm.head(10))




    # Dajemy modelowi kontekst kalendarzowy, ponieważ może zależeć od pory dnia i dnia tygodnia
    df_lgbm["hour"] = df_lgbm.index.hour
    df_lgbm["dayofweek"] = df_lgbm.index.day_of_week

    #print(df_lgbm.head(10))

    r = df_lgbm["suma_opadu_mm"]

    df_lgbm["opad_0h"] = r  # bieżąca godzina

    for k in [2, 3, 6, 12, 24, 48]:                                             #JAK DALEKO W TYŁ MODEL MA PATRZEĆ NA OPADY (godziny)
        df_lgbm[f"opad_{k}h"] = r.rolling(k).sum()

    df_lgbm["opad_max3h"] = r.rolling(3).max()  # intensywność szczytowa

    last_rain = df_lgbm.index.to_series().where(r > 0.1).ffill()
    df_lgbm["h_od_deszczu"] = ((df_lgbm.index.to_series() - last_rain)
                               .dt.total_seconds() / 3600).clip(upper=240)

    #print(df_lgbm.head())

    features = [
        "hour", "dayofweek",
        "opad_0h", "opad_2h", "opad_3h", "opad_6h", "opad_12h",
        "opad_24h", "opad_48h", #"opad_72h", "opad_168h",
        "opad_max3h", "h_od_deszczu"
    ]

    X = df_lgbm[features].copy()
    y = df_lgbm["przepompowano"].copy()

    #print("Liczba cech:", X.shape[1])
    #print("Liczba obserwacji:", len(X))

    #print("\nBraki w cechach:", X.isna().sum())
    #print("\nBraki w y:", y.isna().sum())

    # Usuwamy tylko wiersze z brakującą wartością docelową
    valid_idx = y.notna()

    X = X.loc[valid_idx].copy()
    y = y.loc[valid_idx].copy()

    return X,y


