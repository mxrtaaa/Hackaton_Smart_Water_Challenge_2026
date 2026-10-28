import pandas as pd
from pathlib import Path
from contextlib import redirect_stdout
import sys
if __package__:
    from .data_preparation import convert_meter_data, set_timezone
else:
    from data_preparation import convert_meter_data, set_timezone

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


def chorzow_processed_paths(root):
    """Pliki pomiarów i opadów Chorzowa, bez zależności od aplikacji UI."""
    parent = Path(root) / "data" / "processed"
    folder = next((p for p in parent.iterdir() if p.is_dir() and p.name.casefold() == "chorzów"), None)
    if folder is None:
        raise FileNotFoundError("Brak folderu data/processed/Chorzów.")
    return [folder / name for name in ("scieki.parquet", "energia.parquet", "suma_opadow.parquet")]


def load_processed_chorzow(root):
    """Godzinowe pomiary i ich pochodzenie; braki pozostają nieznane.

    Liczniki narastające przetwarza istniejące convert_meter_data z
    data_preparation. Nie interpolujemy i nie wypełniamy braków zerami.
    """
    root = Path(root)
    wastewater, energy, weather = chorzow_processed_paths(root)
    series, source_info = [], []
    for path, column, output in ((wastewater, "Ilosc_przepompowanych_sciekow", "przepompowano_m3"),
                                 (energy, "Zuzycie_energii_elektrycznej", "energia_kwh")):
        raw = pd.read_parquet(path)
        hourly = convert_meter_data(raw, column, output)
        measured = raw.loc[raw["Sposób_pomiaru"].astype(str).str.strip().eq("Wartość z pomiaru")]
        measured = set_timezone(measured)
        end = measured.index.max()
        # Ostatni przedział nie ma jeszcze następnej pełnej granicy godziny.
        hourly = hourly.loc[hourly.index < end.floor("h")]
        if hourly.dropna().empty:
            raise ValueError(f"Brak kompletnych godzin pomiaru w {path.name}.")
        series.append(hourly)
        source_info.append({"file": str(path.relative_to(root)), "column": column,
                            "rawRows": len(raw), "rawStart": measured.index.min().isoformat(),
                            "rawEnd": end.isoformat(), "lastCompleteHour": hourly.last_valid_index().isoformat(),
                            "granularity": "Odczyty nieregularne; licznik narastający → przyrosty godzinowe."})
    rain = set_timezone(pd.read_parquet(weather))
    rain["suma_opadu_mm"] = pd.to_numeric(rain["suma_opadu_mm"], errors="coerce")
    hourly_rain = rain["suma_opadu_mm"].groupby(level=0).sum(min_count=1).resample("h").sum(min_count=1).rename("opad_mm")
    last = min(s.dropna().index[-1] for s in series)
    frame = pd.concat(series, axis=1).sort_index().asfreq("h").loc[:last]
    frame = frame.join(hourly_rain)
    return frame, {"files": source_info, "weatherFile": str(weather.relative_to(root)),
                   "historyRule": "7/14/30 dni wstecz od ostatniej pełnej godziny wspólnego archiwum. Braki nie są interpolowane.",
                   "excludedLastPartialHour": True}


def weteranow_processed_paths(root):
    """Wyłącznie zatwierdzone źródła Weteranów, bez ZMP i innych lokalizacji."""
    data = Path(root) / "data" / "processed" / "Ruda Śląska"
    return {"hourly": data / "godzinowe-weteranow.parquet",
            "scada": data / "scada-weteranow.parquet",
            "energy": data / "energia-weteranow.parquet"}


def _local_time(value):
    value = pd.Timestamp(value)
    return value.tz_localize("Europe/Warsaw") if value.tzinfo is None else value.tz_convert("Europe/Warsaw")


def load_processed_weteranow(root):
    """Gotowe godzinowe m³/kWh/opady Weteranów i metadane źródła.

    Plik godzinowy powstaje w business_analysis.py. Zawiera już przyrosty,
    więc nie używamy diff() ponownie. load_scada_weterani interpoluje braki,
    dlatego nie jest właściwym źródłem historii rzeczywistych pomiarów.
    Ta funkcja nie wczytuje modeli ani nie uruchamia analizy biznesowej.
    """
    root = Path(root)
    paths = weteranow_processed_paths(root)
    missing = [str(p.relative_to(root)) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Brak zatwierdzonych plików Weteranów: " + ", ".join(missing))
    frame = pd.read_parquet(paths["hourly"], columns=["przepompowano_m3", "energia_kwh", "opad_mm"])
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None or frame.index.has_duplicates:
        raise ValueError("Dane godzinowe Weteranów muszą mieć unikalny indeks czasu ze strefą.")
    frame = frame.sort_index().asfreq("h")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame["opad_mm"].lt(0).any():
        raise ValueError("Historyczne opady Weteranów zawierają ujemne wartości.")
    for column in ("przepompowano_m3", "energia_kwh"):
        frame[column] = frame[column].where(frame[column].ge(0))
    scada_time = pd.read_parquet(paths["scada"], columns=["DataCzas"])["DataCzas"]
    raw_end = _local_time(pd.to_datetime(scada_time).max())
    frame = frame.loc[frame.index < raw_end.floor("h")]
    if frame[["przepompowano_m3", "energia_kwh"]].dropna(how="all").empty:
        raise ValueError("Brak pomiarów w zatwierdzonych danych Weteranów.")
    raw_energy = pd.read_parquet(paths["energy"], columns=["Data_czas", "Wartosc_kWh"])
    numeric_energy = pd.to_numeric(raw_energy["Wartosc_kWh"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    if not numeric_energy.ge(0).any():
        raise ValueError("Plik energii Weteranów nie zawiera poprawnych kWh.")
    return frame, {
        "files": [{"file": str(paths[key].relative_to(root)), "role": role} for key, role in
                  (("hourly", "Godzinowe rzeczywiste m³, kWh i opady — źródło historii"),
                   ("scada", "Kontrola końca pełnego przedziału pomiarów"),
                   ("energy", "Źródłowe godzinowe kWh, bez ponownego różnicowania"))],
        "rawScadaEnd": raw_end.isoformat(),
        "rawEnergyEnd": _local_time(pd.to_datetime(raw_energy["Data_czas"]).max()).isoformat(),
        "historyRule": "7/14/30 dni wstecz od ostatniej pełnej godziny archiwum Weteranów; braków nie interpolujemy.",
    }

def rybnik_processed_paths(root, object_id):
    """Osobne logi i raporty PZ1/PNB1; bez zamieniania obiektów."""
    if object_id not in ("pz1", "pnb1"):
        raise ValueError("Nieznany obiekt Rybnika.")
    data = Path(root) / "data" / "processed" / "Rybnik"
    return {"flow": data / f"{object_id}-2024.parquet",
            "energy": data / f"raport-{object_id}.parquet",
            "weather": data / "suma_opadow.parquet"}


def load_processed_rybnik(root, object_id):
    """Rzeczywiste godziny z logu licznika, raportu kWh i archiwum opadów.

    Korzysta z istniejących funkcji przygotowania finałowych danych. Nie używa
    niekompletnych godzinowe-*-final.parquet ani kodu treningu modeli.
    """
    if __package__:
        from .wczytaj_rybnik_final import ustaw_czas, seria_godzinowa
    else:
        from wczytaj_rybnik_final import ustaw_czas, seria_godzinowa
    root = Path(root)
    paths = rybnik_processed_paths(root, object_id)
    missing = [str(p.relative_to(root)) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Brak plików Rybnika: " + ", ".join(missing))
    raw_flow = pd.read_parquet(paths["flow"])
    raw_energy = pd.read_parquet(paths["energy"])
    expected_meter = {"pz1": "Q1sr_PZ1.2", "pnb1": "Q1sr_PN1"}[object_id]
    flow = raw_flow.loc[raw_flow["Licznik"].astype(str).str.strip().eq(expected_meter)
                        & raw_flow["Sposob_pomiaru"].astype(str).str.strip().eq("Wartość z pomiaru")].copy()
    if flow.empty:
        raise ValueError(f"Brak poprawnych odczytów licznika {expected_meter}.")
    energy = raw_energy.loc[raw_energy["Rodzaj_energii"].astype(str).str.strip().eq("energia czynna pobrana")].copy()
    if "Kompletnosc_procent" in energy:
        energy = energy.loc[pd.to_numeric(energy["Kompletnosc_procent"], errors="coerce").eq(100)]
    # Helper prints diagnostics; stdout of the UI backend is reserved for JSON.
    with redirect_stdout(sys.stderr):
        flow = ustaw_czas(flow)
        flow = flow.sort_index(kind="stable").loc[lambda d: ~d.index.duplicated(keep="first")]
        # Same validity cutoff as models/final/przepompowanie_rybnik_dane.py.
        cutoff = pd.Timestamp("2024-03-19 13:00:00", tz=flow.index.tz)
        flow = flow.loc[flow.index >= cutoff]
        if flow.empty:
            raise ValueError("Brak pomiarów po początku poprawnego okresu licznika.")
        raw_end = flow.index.max()
        pumping = seria_godzinowa(flow, "Wartosc", "licznik", "przepompowano_m3")
        pumping = pumping.loc[pumping.index < raw_end.floor("h")]
        energy = ustaw_czas(energy)
        if energy.empty:
            raise ValueError("Brak kompletnych pomiarów energii czynnej.")
        energy = energy.sort_index(kind="stable").loc[lambda d: ~d.index.duplicated(keep="first")]
        kwh = seria_godzinowa(energy, "Wartosc_kWh_kvar", "godzinowe", "energia_kwh")
        rain = ustaw_czas(pd.read_parquet(paths["weather"]))
    rain["suma_opadu_mm"] = pd.to_numeric(rain["suma_opadu_mm"], errors="coerce")
    if rain["suma_opadu_mm"].lt(0).any():
        raise ValueError("Ujemne opady historyczne Rybnika.")
    rainfall = (rain["suma_opadu_mm"].groupby(level=0).sum(min_count=1)
                .resample("h").sum(min_count=1).rename("opad_mm"))
    if pumping.dropna().empty or kwh.dropna().empty:
        raise ValueError("Brak godzinowych pomiarów m³ lub kWh Rybnika.")
    end = min(pumping.last_valid_index(), kwh.last_valid_index())
    start = max(pumping.index.min(), kwh.index.min())
    frame = pd.concat([pumping, kwh], axis=1).sort_index().asfreq("h").loc[start:end]
    if frame.empty:
        raise ValueError("Brak wspólnego zakresu m³ i kWh Rybnika.")
    frame = frame.join(rainfall)
    return frame, {
        "files": [{"file": str(p.relative_to(root)), "role": role} for p, role in
                  ((paths["flow"], "Narastający licznik m³ → rzeczywiste przyrosty godzinowe"),
                   (paths["energy"], "Raport energii czynnej pobranej, kompletność 100%; gotowe kWh"),
                   (paths["weather"], "Archiwalne opady godzinowe"))],
        "historyRule": "7/14/30 dni wstecz od ostatniej wspólnej pełnej godziny archiwum. Braki pozostają nieznane.",
        "meter": expected_meter, "rawFlowEnd": raw_end.isoformat(),
        "rawEnergyEnd": energy.index.max().isoformat(),
        "validityCutoff": cutoff.isoformat(), "excludedLastPartialHour": True,
        "note": "Źródłem są log i raport obiektu; niekompletne finałowe tabele godzinowe nie są używane.",
    }


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
        "data/raw/Rybnik/Rap_godzinowy_2024-01-01_2024-12-31_423662 - PNB1.csv", 
        encoding="cp1250"
    )


    # Pliki logów z Rybnika (brak nagłówków)

    columns_PNB_PZ = ["Data", "Godzina", "Licznik", "Wartosc", "Sposob_pomiaru"]

    df_ry_pz1_2024 = pd.read_csv(
        "data/raw/Rybnik/PZ1_2024.csv", 
        encoding="utf-8-sig", 
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
        encoding="utf-8-sig", 
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
