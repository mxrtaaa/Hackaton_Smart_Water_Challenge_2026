import pandas as pd
import numpy as np
from sciezka_projektu import get_project_root


def load_scada_weterani(prog_max_weteran=100.0):
    PROJECT_ROOT = get_project_root()
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "Ruda Śląska"

    df_rs_scada_weteran = pd.read_parquet(DATA_PATH / "scada-weteranow.parquet")
    #df_rs_scada_zmp = pd.read_parquet(DATA_PATH / "scada-zmp.parquet")

    '''zadania = [
        (df_rs_scada_weteran, prog_max_weteran),
        (df_rs_scada_zmp, prog_max_zmp)
    ]'''

    '''wynik = []
    for raw_df, prog_max in zadania:'''
    df = df_rs_scada_weteran.copy()
    prog_max=prog_max_weteran

    if "DataCzas" in df.columns:
        df["Data_czas"] = pd.to_datetime(df["DataCzas"])
        df = df.set_index("Data_czas")

    df = df.rename(columns={"LicznikSciekow": "licznik"})

    df["licznik"] = pd.to_numeric(
        df["licznik"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    df = df[df["licznik"].notna()]

    # Agregacja w przypadku zduplikowanych timestampów
    df = df.groupby(level=0).agg({"licznik": "max"})
    df = df.sort_index()

    # Resampling do 1h i uzupełnienie braków licznika w przód
    hourly = df["licznik"].resample("1h").last().replace(0, np.nan).ffill()

    # Wyliczenie przepływu godzinowego
    df_1h = pd.DataFrame({"przepompowano": hourly.diff()})

    # Usunięcie cofnięć licznika (wartości ujemne) oraz pików powyżej progu
    df_1h.loc[(df_1h["przepompowano"] < 0) | (df_1h["przepompowano"] > prog_max), "przepompowano"] = np.nan

    # Interpolacja czasowa i wypełnienie ewentualnych brzegów
    df_1h["przepompowano"] = (
        df_1h["przepompowano"]
        .interpolate(method="time")
        .bfill()
        .ffill()
    )

    return df_1h