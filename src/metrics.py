import pandas as pd

df_ch_przeplyw = pd.read_parquet('data/processed/Chorzów/przeplyw.parquet')
df_ch_scieki = pd.read_parquet('data/processed/Chorzów/scieki.parquet')
df_ch_awarie = pd.read_parquet('data/processed/Chorzów/awarie.parquet')
df_ch_energia = pd.read_parquet('data/processed/Chorzów/energia.parquet')


def create_5min_df(df, field, new_field):
    # field to nazwa pola w oryginalnym dataframe
    # jako że czasem pola są skomplikowane żeby móc je odróżnic w całym pliku csv,
    # a my niekoniecznie do analizy w danym momencie potrzebujemy tych nazw,
    # to new_field to nowa nazwa którą chcemy nadać analizowanemu polu

    # zabezpieczenie typu NA WSZELKI, chociaż to jest robione przy imporcie
    if "Data_czas" in df.columns:
        df["Data_czas"] = pd.to_datetime(df["Data_czas"])
        df = df.set_index("Data_czas")

    # konwersja na liczbowe
    if df[field].dtype == 'object' or df[field].dtype == 'string':
        df[field] = df[field].astype(str).str.replace(',', '.')
    
    df[field] = pd.to_numeric(df[field], errors='coerce')


    # usuwanie zer i śmieci
    df = df[(df[field].notna()) & (df[field] != 0)]

    df_5min = df[field].resample('5min').agg(
        ['mean', 'min', 'max', 'std', 'count']
    )

    df_5min.columns = [f'{new_field}_mean',
                        f'{new_field}_min',
                        f'{new_field}_max',
                        f'{new_field}_std',
                        'liczba_pomiarow']

    df_5min['rolling_mean_1h'] = (
        df_5min[f'{new_field}_mean'].rolling(window=12, min_periods=6).mean()
    )

    df_5min['rolling_min_1h'] = (
        df_5min[f'{new_field}_min'].rolling(window=12, min_periods=6).min()
    )

    df_5min['rolling_max_1h'] = (
        df_5min[f'{new_field}_max'].rolling(window=12, min_periods=6).max()
    )

    df_5min['rolling_std_1h'] = (
        df_5min[f'{new_field}_std'].rolling(window=12, min_periods=6).std()
    )

    return df_5min

print(create_5min_df(df_ch_przeplyw, 'Przeplyw_chwilowy_cieczy', 'przeplyw'))