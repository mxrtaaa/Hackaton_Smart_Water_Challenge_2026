"""Rybnik — wczytanie finałowych danych PZ1 i PNB1 do godzinowych Parquetów.

Plik logu znany z PZ1_2024/PNB1_2024:
    Data;Godzina;Licznik;Wartosc;Sposob_pomiaru    (bez nagłówków)
Raport godzinowy: CSV z nagłówkami albo Parquet z kolumną daty/godziny.
"""

from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd


# ======================= KONFIGURACJA NA FINAŁ =======================

ROOT = Path(__file__).resolve().parents[1]
WEJSCIE = ROOT / "data" / "final"
WYJSCIE = ROOT / "data" / "processed" / "Rybnik"
POGODA = WYJSCIE / "suma_opadow.parquet"

# Tryb "licznik" = stan narastający: ostatni odczyt w godzinie, potem różnica.
# Tryb "godzinowe" = gotowe m³ lub kWh za godzinę: bez różnicowania.
# W logu bez nagłówków kolumna wartości nazywa się "Wartosc".
# Jeżeli w logu występuje kilka nazw liczników, wpisujemy właściwą w "licznik_nazwa".

columns = ["Licznik", "Data_czas", "Wartosc", "Sposob_pomiaru"]

df_pz1_1 = pd.read_csv(ROOT / "data" / "raw" / "Rybnik" / "PZ1_2024.txt", sep='\t', header=None, names=columns)
df_pz1_2 = pd.read_csv(WEJSCIE / "PZ1_ 2025.csv", sep=';', header=None, names=["Licznik", "Data_czas", "Wartosc"])
df_pz1_3 = pd.read_csv(WEJSCIE / "PZ1_2026.csv", sep=';', header=None, names=columns)
df_pz1 = pd.concat([df_pz1_1, df_pz1_2, df_pz1_3], ignore_index=True)
df_pz1.to_csv(WEJSCIE / "PZ1_all.csv")

df_pnb1 = pd.read_csv(WEJSCIE / "PNB1.txt", header=None, names=columns, sep='\t')
df_pnb1.to_csv(WEJSCIE / "PNB1_all.csv")

df_energia_pnb1_1 = pd.read_csv(ROOT / "data" / "raw" / "Rybnik" / "Rap_godzinowy_2024-01-01_2024-12-31_423662 - PNB1.csv", encoding="cp1250", sep=';')
df_energia_pnb1_2 = pd.read_csv(WEJSCIE / "Rap_godzinowy_2025-01-01_2025-12-31_423660 - PNB1.csv", sep=';', encoding="cp1250")
df_energia_pnb1_3 = pd.read_csv(WEJSCIE / "Rap_godzinowy_2026-01-01_2026-09-28_423667 - PNB1.csv", sep=';', encoding="cp1250")


df_energia_pnb1 = pd.concat([df_energia_pnb1_1, 
                             df_energia_pnb1_2, 
                             df_energia_pnb1_3], ignore_index=True)
df_energia_pnb1.to_csv(
    WEJSCIE / "energia_PNB1_all.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)


df_energia_pz1_1 = pd.read_csv(ROOT / "data" / "raw" / "Rybnik" / "Rap_godzinowy_2023-10-01_2024-09-30_423634 - PZ1.csv", sep=';', encoding="cp1250")
df_energia_pz1_2 = pd.read_csv(WEJSCIE / "Rap_godzinowy_2024-09-28_2025-10-01_423628 - PZ1.csv", sep=';', encoding="cp1250")
df_energia_pz1 = pd.concat([df_energia_pz1_1,
                             df_energia_pz1_2], ignore_index=True)
df_energia_pz1.to_csv(
    WEJSCIE / "energia_PZ1_all.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)


STACJE = {
    "PZ1": {
        "przeplyw_plik": WEJSCIE / "PZ1_all.csv",
        "przeplyw_format": "log",       # "log" lub "raport"
        "przeplyw_tryb": "licznik",    # "licznik" lub "godzinowe"
        "przeplyw_kolumna": "Wartosc",
        "licznik_nazwa": 'Q1sr_PZ1.2',
        "energia_plik": WEJSCIE / "energia_PZ1_all.csv",
        "energia_tryb": "godzinowe",
        "energia_kolumna": "Wartosc[kWh/kvar]",
    },
    "PNB1": {
        "przeplyw_plik": WEJSCIE / "PNB1_all.csv",
        "przeplyw_format": "log",
        "przeplyw_tryb": "licznik",
        "przeplyw_kolumna": "Wartosc",
        "licznik_nazwa": "Q1sr_PN1",
        "energia_plik": WEJSCIE / "energia_PNB1_all.csv",
        "energia_tryb": "godzinowe",
        "energia_kolumna": "Wartosc[kWh/kvar]",
    },
}

# Daty potwierdzonych błędów pomiaru (jeśli występują w nowym pliku).
# np "PZ1": [("2024-03-05 00:00", "2024-03-20 23:00")]
WYKLUCZONE_OKRESY = {"PZ1": [("2024-09-28 00:00", "2024-09-30 00:00"), ("2024-01-01", "2024-03-19 14:00")], "PNB1": []}

STREFA = timezone(timedelta(hours=1))
LOG_KOLUMNY = ["Licznik", "Data_czas", "Wartosc", "Sposob_pomiaru"]

# ====================== PRZETWARZANIE DANYCH =========================


LOG_KOLUMNY = [
    "Licznik", "Data_czas", "Wartosc", "Sposob_pomiaru"
]


def wczytaj_log_surowy(plik):
    # Rozpoznajemy separator na podstawie pierwszego wiersza.
    with open(plik, encoding="utf-8-sig") as f:
        pierwszy = f.readline()

    separator = (
        "\t" if "\t" in pierwszy else
        ";" if ";" in pierwszy else
        ","
    )

    df = pd.read_csv(
        plik,
        sep=separator,
        header=None,
        dtype=str,
        encoding="utf-8-sig",
        low_memory=False
    )

    if df.shape[1] not in (3, 4):
        raise ValueError(
            f"{plik.name}: znaleziono {df.shape[1]} kolumn "
            "zamiast 3 lub 4."
        )

    if df.shape[1] == 3:
        df["brakujaca_kolumna"] = pd.NA

    df.columns = LOG_KOLUMNY

    # Pomijamy nagłówek, jeśli występuje w pliku.
    df = df.loc[
        df["Licznik"].str.strip().ne("Licznik")
    ].copy()

    print(f"{plik.name}: {len(df)} wierszy")
    return df


# PRZEPŁYW PZ1 — 2024, 2025, 2026

df_pz1 = pd.concat([
    wczytaj_log_surowy(
        ROOT / "data" / "raw" / "Rybnik" / "PZ1_2024.txt"
    ),
    wczytaj_log_surowy(WEJSCIE / "PZ1_ 2025.csv"),
    wczytaj_log_surowy(WEJSCIE / "PZ1_2026.csv")
], ignore_index=True)

df_pz1.to_csv(
    WEJSCIE / "PZ1_all.csv",
    sep=";",
    index=False,
    header=False
)

# PRZEPŁYW PNB1

df_pnb1 = wczytaj_log_surowy(WEJSCIE / "PNB1.txt")

df_pnb1.to_csv(
    WEJSCIE / "PNB1_all.csv",
    sep=";",
    index=False,
    header=False
)


def wczytaj(plik, format_pliku="raport"):
    """Wczytuje CSV/TXT (średnik) lub Parquet, z komunikatem o brakach."""
    if not plik.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {plik}")
    if plik.suffix.lower() == ".parquet":
        return pd.read_parquet(plik)
    if plik.suffix.lower() not in {".csv", ".txt"}:
        raise ValueError(f"Nieobsługiwany typ pliku: {plik.suffix}")

    for kodowanie in ("utf-8-sig", "cp1250"):
        try:
            df = pd.read_csv(
                plik, sep=";", encoding=kodowanie,
                header=None if format_pliku == "log" else 0,
                low_memory=False,
            )
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Nie udało się odczytać kodowania: {plik}")

    if format_pliku == "log":
        if df.shape[1] != 4:
            raise ValueError(
                f"{plik.name}: oczekiwano 4 kolumn, "
                f"otrzymano {df.shape[1]}"
            )

        df.columns = LOG_KOLUMNY

        df = df.loc[
            df["Licznik"].astype(str).str.strip().ne("Licznik")
        ].copy()

    return df



def ustaw_czas(df):
    """Różne formaty dat -> godziny w stałym UTC+1."""

    if "Data_czas" in df.columns:
        surowe_daty = df["Data_czas"]

    elif "DataCzas" in df.columns:
        surowe_daty = df["DataCzas"]

    elif "Data i godzina" in df.columns:
        surowe_daty = df["Data i godzina"]

    elif {"Data", "Godzina"}.issubset(df.columns):
        surowe_daty = (
            df["Data"].astype(str) + " " +
            df["Godzina"].astype(str)
        )

    elif {"Data", "Czas"}.issubset(df.columns):
        surowe_daty = (
            df["Data"].astype(str) + " " +
            df["Czas"].astype(str)
        )

    else:
        raise ValueError(
            f"Nie rozpoznaję kolumn czasu: {list(df.columns)}"
        )

    surowe_daty = surowe_daty.astype("string").str.strip()

    print(
        f"  Parsowanie {len(surowe_daty):,} dat...",
        flush=True
    )

    # Najpierw szybko próbujemy format ISO,
    # np. 2025-03-17 12:30:00
    daty = pd.to_datetime(
        surowe_daty,
        format="ISO8601",
        errors="coerce"
    )

    # Inny częsty format:
    # 17.03.2025 12:30:00
    brak = daty.isna() & surowe_daty.notna()

    if brak.any():
        daty.loc[brak] = pd.to_datetime(
            surowe_daty.loc[brak],
            format="%d.%m.%Y %H:%M:%S",
            errors="coerce"
        )

    # Dopiero pozostałe, nietypowe daty
    # próbujemy rozpoznawać automatycznie.
    brak = daty.isna() & surowe_daty.notna()

    if brak.any():
        daty.loc[brak] = pd.to_datetime(
            surowe_daty.loc[brak],
            format="mixed",
            dayfirst=True,
            errors="coerce"
        )

    print(
        "  Daty rozpoznane:",
        int(daty.notna().sum()),
        "/",
        len(daty),
        flush=True
    )


    print("  Liczba dat według lat PRZED zmianą strefy:")
    print(daty.dt.year.value_counts(dropna=False).sort_index())

    indeks = pd.DatetimeIndex(daty)

    if indeks.tz is None:
        indeks = indeks.tz_localize(
            "Europe/Warsaw",
            ambiguous="NaT",
            nonexistent="NaT"
        )

    indeks = indeks.tz_convert(STREFA)

    print("  Odrzucone daty:", int(indeks.isna().sum()))

    df = df.copy()
    df.index = indeks
    df = df.loc[df.index.notna()].copy()
    df.index.name = "Data_czas"

    return df



def liczby(df, kolumna):
    if not kolumna or kolumna not in df.columns:
        raise ValueError(
            f"Nie ma kolumny '{kolumna}'. Kolumny pliku: {list(df.columns)}"
        )
    return pd.to_numeric(
        df[kolumna].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def seria_godzinowa(df, kolumna, tryb, wynik):
    """Przekształca pojedynczy pomiar na godziny; bez interpolacji i zer zastępczych."""
    wartosci = liczby(df, kolumna)
    wartosci = wartosci.where(wartosci >= 0)
    wartosci = wartosci.sort_index(kind="stable")
    # Jeden odczyt na znacznik czasu, w kolejności z pliku.
    wartosci = wartosci.loc[~wartosci.index.duplicated(keep="last")]
    n_godzin_z_duplikatami = int((wartosci.resample("h").size() > 1).sum())
    if n_godzin_z_duplikatami:
        print(f"  Godziny z kilkoma pomiarami: {n_godzin_z_duplikatami}")

    h = wartosci.resample("h").last() if tryb == "licznik" else wartosci.resample("h").first()
    if tryb == "licznik":
        h = h.diff().where(lambda s: s >= 0)
    elif tryb != "godzinowe":
        raise ValueError(f"Nieznany tryb '{tryb}'; użyj 'licznik' lub 'godzinowe'")
    return h.rename(wynik)


def przeplyw(stacja, cfg):
    df = wczytaj(cfg["przeplyw_plik"], cfg["przeplyw_format"])
    if cfg["przeplyw_format"] == "log":
        nazwy = df["Licznik"].astype(str).str.strip()
        dostepne = sorted(nazwy.loc[df["Licznik"].notna()].unique())
        wybor = cfg["licznik_nazwa"]
        if wybor is None:
            if len(dostepne) != 1:
                raise ValueError(
                    f"{stacja}: kilka liczników w logu: {dostepne}. "
                    "Wpisz właściwy w KONFIGURACJI: licznik_nazwa."
                )
            wybor = dostepne[0]
        if wybor not in dostepne:
            raise ValueError(f"{stacja}: brak licznika '{wybor}'; dostępne: {dostepne}")
        df = df.loc[nazwy.eq(wybor)].copy()
        if "Sposob_pomiaru" in df:
            pomiary = df["Sposob_pomiaru"].astype("string").str.strip()
            poprawne = pomiary.isna() | pomiary.eq("Wartość z pomiaru")
            df = df.loc[poprawne].copy()
            print("Rekordy po filtrze sposobu pomiaru:", len(df))

        print(f"  Wybrany licznik przepływu: {wybor}")

    df = ustaw_czas(df)
    print("  Przykładowe surowe wartości m³:",
          liczby(df, cfg["przeplyw_kolumna"]).dropna().head(5).tolist())
    seria = seria_godzinowa(
        df, cfg["przeplyw_kolumna"], cfg["przeplyw_tryb"], "przepompowano_m3"
    )

    return seria


def energia(cfg):
    plik = cfg["energia_plik"]
    if plik is None or not plik.exists():
        print("  Brak pliku energii — pozostawiam kWh jako NaN.")
        return None
    df = wczytaj(plik)
    if not cfg["energia_kolumna"]:
        raise ValueError(
            f"W pliku {plik.name} wybierz 'energia_kolumna'. "
            f"Dostępne kolumny: {list(df.columns)}"
        )
    df = ustaw_czas(df)

    return seria_godzinowa(df, cfg["energia_kolumna"], cfg["energia_tryb"], "energia_kwh")


# =========================== URUCHOMIENIE ============================

if __name__ == "__main__":
    WYJSCIE.mkdir(parents=True, exist_ok=True)
    print("Dane finałowe:", WEJSCIE)

    opady = None
    if POGODA.exists():
        pogoda = ustaw_czas(pd.read_parquet(POGODA))
        opady = seria_godzinowa(pogoda, "suma_opadu_mm", "godzinowe", "opad_mm")
        
    else:
        print("UWAGA: brak lokalnych opadów; kolumna opad_mm pozostanie NaN.")

    for stacja, cfg in STACJE.items():
        print(f"\n========== RYBNIK: {stacja} ==========")
        if not cfg["przeplyw_plik"].exists():
            print(f"Pominięto: brak {cfg['przeplyw_plik'].name}")
            continue

        m3 = przeplyw(stacja, cfg)
        kwh = energia(cfg)
        serie = [m3] + ([kwh] if kwh is not None else [])
        df = pd.concat(serie, axis=1).sort_index().asfreq("h")
        if "energia_kwh" not in df:
            df["energia_kwh"] = float("nan")
        df["opad_mm"] = opady.reindex(df.index) if opady is not None else float("nan")
        df = df[["przepompowano_m3", "energia_kwh", "opad_mm"]]

        if df.index.has_duplicates or df.index.tz is None:
            raise ValueError(f"{stacja}: niepoprawny indeks godzinowy")

        print("Zakres:", df.index.min(), "—", df.index.max())
        print("Godziny:", len(df))
        print("Braki (nie są zamieniane na zera):\n", df.isna().sum())
        print("Kwantyle m³/h:\n", df["przepompowano_m3"].quantile([0.5, 0.95, 0.99, 0.999]))
        print("Największe przyrosty m³/h:\n", df["przepompowano_m3"].nlargest(5))
        if df["opad_mm"].isna().any():
            print("UWAGA: niekompletny opad. Uzupełnij pogodę PRZED treningiem modelu opadowego.")

        for start, koniec in WYKLUCZONE_OKRESY[stacja]:
            a = pd.Timestamp(start).tz_localize("Europe/Warsaw").tz_convert(STREFA)
            b = pd.Timestamp(koniec).tz_localize("Europe/Warsaw").tz_convert(STREFA)

            do_usuniecia = (df.index >= a) & (df.index <= b + pd.Timedelta(hours=1))
            print(f"Usunięte godziny ({stacja}):", do_usuniecia.sum())
            df = df.loc[~do_usuniecia].copy()
        
        zapis = WYJSCIE / f"godzinowe-{stacja.lower()}-final.parquet"
        df.to_parquet(zapis)
        print("Zapisano:", zapis)


    df = pd.read_parquet(
        WYJSCIE / "godzinowe-pz1-final.parquet"
    )

    print("\n========== KONTROLA PZ1 ==========")

    print("Zakres:", df.index.min(), "—", df.index.max())

    print("\nLiczba poprawnych pomiarów według lat:")
    print(
        df.groupby(df.index.year)[
            ["przepompowano_m3", "energia_kwh", "opad_mm"]
        ].count()
    )

    print("\nOstatnie poprawne pomiary:")
    for kolumna in ["przepompowano_m3", "energia_kwh", "opad_mm"]:
        print(kolumna, ":", df[kolumna].last_valid_index())

    print(df)
