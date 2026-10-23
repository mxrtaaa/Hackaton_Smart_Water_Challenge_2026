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

df_ch_przeplyw['Data_czas'] = pd.to_datetime(df_ch_przeplyw['Data'].astype(str) + ' ' + df_ch_przeplyw['Godzina'].astype(str), format='%d.%m.%Y %H:%M.%S')
df_ch_scieki['Data_czas'] = pd.to_datetime(df_ch_scieki['Data'].astype(str) + ' ' + df_ch_scieki['Godzina'].astype(str), format='%d.%m.%Y %H:%M:%S')
df_ch_awarie['Data_czas'] = pd.to_datetime(df_ch_awarie['Data'].astype(str) + ' ' + df_ch_awarie['Czas'].astype(str), format='%d.%m.%Y %H:%M:%S')
df_ch_energia['Data_czas'] = pd.to_datetime(df_ch_energia['Data'].astype(str) + ' ' + df_ch_energia['Czas'].astype(str), format='%d.%m.%Y %H:%M:%S')

df_ch_przeplyw.to_parquet('data/processed/Chorzów/przeplyw.parquet')
df_ch_scieki.to_parquet('data/processed/Chorzów/scieki.parquet')
df_ch_awarie.to_parquet('data/processed/Chorzów/awarie.parquet')
df_ch_energia.to_parquet('data/processed/Chorzów/energia.parquet')