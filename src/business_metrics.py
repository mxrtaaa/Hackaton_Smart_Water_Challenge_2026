import pandas as pd
import numpy as np
from pathlib import Path
from data_preparation import convert_meter_data

DATA_PATH = Path("./data/processed/Chorzów")

df_ch_opady = pd.read_parquet(DATA_PATH / "suma_opadow.parquet")
df_ch_przeplyw = pd.read_parquet(DATA_PATH / "przeplyw.parquet")
df_ch_energia = pd.read_parquet(DATA_PATH / "energia.parquet")
df_ch_scieki = pd.read_parquet(DATA_PATH / "scieki.parquet")

CENA_ENERGII = 1.0  # zł/kWh

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
