import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_ch_przeplyw = pd.read_csv(
    "data/raw/Chorzów/Dane do konkursu/Dane przepływowe.csv",
    encoding="cp1250",
    sep=";")
df_ch_scieki = pd.read_csv(
    "data/raw/Chorzów/Dane do konkursu/Ilość przepompowanych ścieków.csv",
    encoding="utf-8-sig",
    sep=";")
df_ch_awarie = pd.read_csv(
    "data/raw/Chorzów/Dane do konkursu/Wszystkie Awarie.csv",
    encoding="cp1250",
    sep=";")
df_ch_energia = pd.read_csv(
    "data/raw/Chorzów/Dane do konkursu/Zużycie energii elektrycznej.csv",
    encoding="utf-8-sig",
    sep=";")

df_rs_energia_weteranow = pd.read_csv(
    "data/raw/Ruda Sląska/energia 2024-2026 - Weteranów.csv",
    encoding="utf-8-sig",
    sep=";"
)

df_rs_energia_zmp = pd.read_csv(
    "data/raw/Ruda Sląska/energia 2024-2026 - ZMP.csv",
    encoding="utf-8-sig",
    sep=";"
)

df_rs_intensywnosc_opadu = pd.read_csv(
    "data/raw/Ruda Sląska/Intensywnosc opadu.csv",
    encoding="utf-8-sig",
    sep=";"
)

df_rs_suma_opadu = pd.read_csv(
    "data/raw/Ruda Sląska/Suma opadu.csv",
    encoding="utf-8-sig",
    sep=";"
)

columns_scada = [
    "Stempel", "DataCzas", "PoziomSciekow", "PrzeplywSciekow", 
    "PradP1", "PradP2", "LicznikSciekow", "IloscZalaczenP1", 
    "IloscZalaczenP2", "CzasPracyP1", "CzasPracyP2", 
    "PoziomMin", "PoziomMax", "PoziomOn", "PoziomOff"
]

df_rs_scada_weteranow_2024 = pd.read_csv(
    "data/raw/Ruda Sląska/SCADA_weteranow_2024.csv",
    encoding="utf-8-sig",
    sep=";",
    header=None,
    names=columns_scada
)

df_rs_scada_weteranow_2025 = pd.read_csv(
    "data/raw/Ruda Sląska/SCADA_weteranow_2025.csv",
    encoding="utf-8-sig",
    sep=";",
    header=None,
    names=columns_scada
)

df_rs_scada_weteranow_2026 = pd.read_csv(
    "data/raw/Ruda Sląska/SCADA_weteranow_2026.csv",
    encoding="utf-8-sig",
    sep=";",
    header=None,
    names=columns_scada
)

df_rs_scada_weteranow = pd.concat([df_rs_scada_weteranow_2024, df_rs_scada_weteranow_2025, df_rs_scada_weteranow_2026])

df_rs_scada_zmp_2024 = pd.read_csv(
    "data/raw/Ruda Sląska/SCADA_zmp_2024.csv",
    encoding="utf-8-sig",
    sep=";",
    header=None,
    names=columns_scada
)

df_rs_scada_zmp_2025 = pd.read_csv(
    "data/raw/Ruda Sląska/SCADA_zmp_2025.csv",
    encoding="utf-8-sig",
    sep=";",
    header=None,
    names=columns_scada
)

df_rs_scada_zmp_2026 = pd.read_csv(
    "data/raw/Ruda Sląska/SCADA_zmp_2026.csv",
    encoding="utf-8-sig",
    sep=";",
    header=None,
    names=columns_scada
)

df_rs_scada_zmp = pd.concat([df_rs_scada_zmp_2024, df_rs_scada_zmp_2025, df_rs_scada_zmp_2026])



df_ch_przeplyw['Data_czas'] = pd.to_datetime(df_ch_przeplyw['Data'].astype(str) + ' ' + df_ch_przeplyw['Godzina'].astype(str), format='%d.%m.%Y %H:%M.%S')
df_ch_scieki['Data_czas'] = pd.to_datetime(df_ch_scieki['Data'].astype(str) + ' ' + df_ch_scieki['Godzina'].astype(str), format='%d.%m.%Y %H:%M:%S')
df_ch_awarie['Data_czas'] = pd.to_datetime(df_ch_awarie['Data'].astype(str) + ' ' + df_ch_awarie['Czas'].astype(str), format='%d.%m.%Y %H:%M:%S')
df_ch_energia['Data_czas'] = pd.to_datetime(df_ch_energia['Data'].astype(str) + ' ' + df_ch_energia['Czas'].astype(str), format='%d.%m.%Y %H:%M:%S')
df_rs_energia_weteranow['Data_czas'] = pd.to_datetime(df_rs_energia_weteranow['Data'].astype(str) + ' ' + df_rs_energia_weteranow['Godzina'].astype(str), format='%d.%m.%Y %H:%M:%S')
df_rs_energia_zmp['Data_czas'] = pd.to_datetime(df_rs_energia_zmp['Data'].astype(str) + ' ' + df_rs_energia_zmp['Godzina'].astype(str), format='%d.%m.%Y %H:%M:%S')
df_rs_intensywnosc_opadu['Data_czas'] = pd.to_datetime(df_rs_intensywnosc_opadu['Data'].astype(str) + ' ' + df_rs_intensywnosc_opadu['Godzina'].astype(str), format='%d.%m.%Y %H:%M:%S')
df_rs_suma_opadu['Data_czas'] = pd.to_datetime(df_rs_suma_opadu['Data'].astype(str) + ' ' + df_rs_suma_opadu['Godzina'].astype(str), format='%d.%m.%Y %H:%M:%S')
df_rs_scada_weteranow['DataCzas'] = pd.to_datetime(df_rs_scada_weteranow['DataCzas'].astype(str), format='%Y-%m-%d %H:%M:%S.%f')
df_rs_scada_zmp['DataCzas'] = pd.to_datetime(df_rs_scada_zmp['DataCzas'].astype(str), format='%Y-%m-%d %H:%M:%S.%f')

df_ch_przeplyw.to_parquet('data/processed/Chorzów/przeplyw.parquet')
df_ch_scieki.to_parquet('data/processed/Chorzów/scieki.parquet')
df_ch_awarie.to_parquet('data/processed/Chorzów/awarie.parquet')
df_ch_energia.to_parquet('data/processed/Chorzów/energia.parquet')
df_rs_energia_weteranow.to_parquet('data/processed/Ruda Śląska/energia-weteranow.parquet')
df_rs_energia_zmp.to_parquet('data/processed/Ruda Śląska/energia-zmp.parquet')
df_rs_intensywnosc_opadu.to_parquet('data/processed/Ruda Śląska/intensywnosc-opadu.parquet')
df_rs_suma_opadu.to_parquet('data/processed/Ruda Śląska/suma-opadu.parquet')
df_rs_scada_weteranow.to_parquet('data/processed/Ruda Śląska/scada-weteranow.parquet')
df_rs_scada_zmp.to_parquet('data/processed/Ruda Śląska/scada-zmp.parquet')