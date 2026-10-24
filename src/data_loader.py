import pandas as pd

def load_standard_file(filepath, encoding="utf-8-sig", time_column="Godzina", date_format="%d.%m.%Y %H:%M:%S"):
    df = pd.read_csv(filepath, encoding=encoding, sep=";")
    df['Data_czas'] = pd.to_datetime(
        df['Data'].astype(str) + ' ' + df[time_column].astype(str), 
        format=date_format
    )
    return df

def load_scada_files(base_dir, station, years, columns):
    dfs = []
    for year in years:
        filepath = f"{base_dir}/SCADA_{station}_{year}.csv"
        df = pd.read_csv(filepath, encoding="utf-8-sig", sep=";", header=None, names=columns)
        dfs.append(df)
        
    df_combined = pd.concat(dfs, ignore_index=True)
    df_combined['DataCzas'] = pd.to_datetime(df_combined['DataCzas'].astype(str), format='%Y-%m-%d %H:%M:%S.%f')
    return df_combined

if __name__ == "__main__":
    
    # Wczytywanie plików standardowych - Chorzów i Ruda Śląska

    df_ch_przeplyw = load_standard_file(
        "data/raw/Chorzów/Dane do konkursu/Dane przepływowe.csv", 
        encoding="cp1250", 
        date_format='%d.%m.%Y %H:%M.%S'
    )
    
    df_ch_scieki = load_standard_file("data/raw/Chorzów/Dane do konkursu/Ilość przepompowanych ścieków.csv")
    
    df_ch_awarie = load_standard_file(
        "data/raw/Chorzów/Dane do konkursu/Wszystkie Awarie.csv", 
        encoding="cp1250", 
        time_column="Czas"
    )
    
    df_ch_energia = load_standard_file(
        "data/raw/Chorzów/Dane do konkursu/Zużycie energii elektrycznej.csv", 
        time_column="Czas"
    )
    
    df_rs_energia_weteranow = load_standard_file("data/raw/Ruda Sląska/energia 2024-2026 - Weteranów.csv")
    df_rs_energia_zmp = load_standard_file("data/raw/Ruda Sląska/energia 2024-2026 - ZMP.csv")
    df_rs_intensywnosc_opadu = load_standard_file("data/raw/Ruda Sląska/Intensywnosc opadu.csv")
    df_rs_suma_opadu = load_standard_file("data/raw/Ruda Sląska/Suma opadu.csv")


    # Wczytywanie i łączenie plików SCADA - Ruda Śląska

    columns_scada = [
        "Stempel", "DataCzas", "PoziomSciekow", "PrzeplywSciekow", 
        "PradP1", "PradP2", "LicznikSciekow", "IloscZalaczenP1", 
        "IloscZalaczenP2", "CzasPracyP1", "CzasPracyP2", 
        "PoziomMin", "PoziomMax", "PoziomOn", "PoziomOff"
    ]
    
    lata_scada = [2024, 2025, 2026]
    baza_rs = "data/raw/Ruda Sląska"
    
    df_rs_scada_weteranow = load_scada_files(baza_rs, "weteranow", lata_scada, columns_scada)
    df_rs_scada_zmp = load_scada_files(baza_rs, "zmp", lata_scada, columns_scada)


    # Raporty godzinowe z Rybnika

    df_ry_raport_pz1_1 = load_standard_file(
        "data/raw/Rybnik/Rap_godzinowy_2023-10-01_2024-09-30_423634 - PZ1.csv", 
        encoding="cp1250"
    )
    df_ry_raport_pz1_2 = load_standard_file(
        "data/raw/Rybnik/Rap_godzinowy_2024-09-30_2025-09-30_423633 - PZ1.csv", 
        encoding="cp1250"
    )
    df_ry_raport_pz1 = pd.concat([df_ry_raport_pz1_1, df_ry_raport_pz1_2], ignore_index=True)

    df_ry_raport_pnb1 = load_standard_file(
        "data/raw/Rybnik/Rap_godzinowy_2024-01-01_2024-12-31_423634 - PNB1.csv", 
        encoding="cp1250"
    )


    # Pliki logów z Rybnika (brak nagłówków)

    columns_PNB_PZ = ["Data", "Godzina", "Licznik", "Wartosc", "Sposob_pomiaru"]

    df_ry_pz1_2024 = pd.read_csv(
        "data/raw/Rybnik/PZ1_2024.csv", 
        encoding="cp1250", 
        sep=";", 
        header=None, 
        names=columns_PNB_PZ
    )
    df_ry_pz1_2024['Data_czas'] = pd.to_datetime(
        df_ry_pz1_2024['Data'].astype(str) + ' ' + df_ry_pz1_2024['Godzina'].astype(str), 
        format="%d.%m.%Y %H:%M:%S"
    )

    df_ry_pnb1_2024 = pd.read_csv(
        "data/raw/Rybnik/PNB1_2024.csv", 
        encoding="cp1250", 
        sep=";", 
        header=None, 
        names=columns_PNB_PZ
    )
    df_ry_pnb1_2024['Data_czas'] = pd.to_datetime(
        df_ry_pnb1_2024['Data'].astype(str) + ' ' + df_ry_pnb1_2024['Godzina'].astype(str), 
        format="%d.%m.%Y %H:%M:%S"
    )


    # Zapis do plików Parquet

    slownik_eksportu = {
        'data/processed/Chorzów/przeplyw.parquet': df_ch_przeplyw,
        'data/processed/Chorzów/scieki.parquet': df_ch_scieki,
        'data/processed/Chorzów/awarie.parquet': df_ch_awarie,
        'data/processed/Chorzów/energia.parquet': df_ch_energia,
        'data/processed/Ruda Śląska/energia-weteranow.parquet': df_rs_energia_weteranow,
        'data/processed/Ruda Śląska/energia-zmp.parquet': df_rs_energia_zmp,
        'data/processed/Ruda Śląska/intensywnosc-opadu.parquet': df_rs_intensywnosc_opadu,
        'data/processed/Ruda Śląska/suma-opadu.parquet': df_rs_suma_opadu,
        'data/processed/Ruda Śląska/scada-weteranow.parquet': df_rs_scada_weteranow,
        'data/processed/Ruda Śląska/scada-zmp.parquet': df_rs_scada_zmp,
        'data/processed/Rybnik/raport-pz1.parquet': df_ry_raport_pz1,
        'data/processed/Rybnik/raport-pnb1.parquet': df_ry_raport_pnb1,
        'data/processed/Rybnik/pz1-2024.parquet': df_ry_pz1_2024,
        'data/processed/Rybnik/pnb1-2024.parquet': df_ry_pnb1_2024
    }

    for sciezka, df in slownik_eksportu.items():
        katalog = Path(sciezka).parent
        katalog.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(sciezka)