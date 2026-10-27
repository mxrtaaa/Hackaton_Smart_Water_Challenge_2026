import pandas as pd
import numpy as np
import sys 
from pathlib import Path

src_path = Path(__file__).resolve().parents[2] / 'src'
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from sciezka_projektu import get_project_root
from data_preparation import set_timezone


def przygotuj_dane_przepompowania_z_opadami():
    PROJECT_ROOT = get_project_root()

    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "Chorzów"

    df_ch_scieki = pd.read_parquet(DATA_PATH / 'scieki.parquet')
    df_ch_opady = pd.read_parquet(DATA_PATH / 'suma_opadow.parquet')


    df = df_ch_scieki.copy()

    df["Sposób_pomiaru"] = df["Sposób_pomiaru"].astype(str).str.strip()
    df = df[df["Sposób_pomiaru"] == "Wartość z pomiaru"].copy()

    
    df = set_timezone(df)

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


    df_ch_opady["Data_czas"] = pd.to_datetime(df_ch_opady["Data_czas"])
    df_ch_opady = df_ch_opady.set_index("Data_czas")

    fixed_tz = timezone(timedelta(hours=1))

    if df_ch_opady.index.tz is None:
        # Starszy plik opadów przechowuje polski czas lokalny
        # bez informacji o strefie.
        df_ch_opady.index = (
            df_ch_opady.index
            .tz_localize(
                "Europe/Warsaw",
                ambiguous="infer",
                nonexistent="raise"
            )
            .tz_convert(fixed_tz)
        )
    else:
        # Nowszy plik ma już przypisaną strefę.
        df_ch_opady.index = df_ch_opady.index.tz_convert(fixed_tz)

    df_ch_opady = df_ch_opady.sort_index()

    df_ch_opady = df_ch_opady.groupby(level=0).agg(
        {"suma_opadu_mm": "sum"}
    )

    print("Strefa przepływu:", df_1h.index.tz if "df_1h" in locals() else df.index.tz)
    print("Strefa opadów:", df_ch_opady.index.tz)


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
    df_lgbm["month"] = df_lgbm.index.month 
    df_lgbm["weekend"] = (df_lgbm.index.day_of_week >= 5).astype(int)

    #Żeby mieć cykliczność opadów
    df_lgbm["hour_sin"] = np.sin(2 * np.pi * df_lgbm.index.hour / 24)
    df_lgbm["hour_cos"] = np.cos(2 * np.pi * df_lgbm.index.hour / 24)

    #print(df_lgbm.head(10))

    r = df_lgbm["suma_opadu_mm"]

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

    #historia przepompowań
    for k in [1, 3, 6, 24]:                                            
        df_lgbm[f"przepompowano_lag_{k}h"] = df_lgbm[f"przepompowano"].shift(k)

    df_lgbm[f"przepompowano_mean_6h"] = df_lgbm["przepompowano"].shift(1).rolling(6, min_periods=6).mean()

    features = [
        # Kalendarzowe
        "hour", "dayofweek", "month", "weekend",
        "hour_sin", "hour_cos",

        # Opady
        "opad_0h", "opad_2h", "opad_3h", "opad_6h",
        "opad_12h", "opad_24h", "opad_48h",
        "opad_max3h", "h_od_deszczu",

        # Opóźnienia opadów
        "opad_lag_2h",
        "opad_lag_3h", "opad_lag_6h",
        "opad_lag_12h", "opad_lag_24h",
        "opad_lag_48h",

        # Historia przepompowań
        #"przepompowano_lag_1h",
        #"przepompowano_lag_3h",
        #"przepompowano_lag_6h",
        #"przepompowano_lag_24h",
        #"przepompowano_mean_6h"
    ]

    X = df_lgbm[features].copy()
    y = df_lgbm["przepompowano"].copy()

    #print("Liczba cech:", X.shape[1])
    #print("Liczba obserwacji:", len(X))

    #print("\nBraki w cechach:", X.isna().sum())
    #print("\nBraki w y:", y.isna().sum())

    # Usuwamy tylko wiersze z brakującą wartością docelową
    first_valid = y.first_valid_index()
    X = X.loc[first_valid:].copy()
    y = y.loc[first_valid:].copy()

    # Jeśli w środku były luki, wypełnij je interpolacją:
    if y.isna().any():
        y = y.interpolate(method="time")

    # Wymuszenie metadanej częstotliwości w indeksie
    y.index.freq = pd.infer_freq(y.index) or "1h"
    X.index.freq = y.index.freq

    return X,y


