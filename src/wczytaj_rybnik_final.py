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
POGODA = WEJSCIE / "suma_opadow.parquet"

# Tryb "licznik" = stan narastający: ostatni odczyt w godzinie, potem różnica.
# Tryb "godzinowe" = gotowe m³ lub kWh za godzinę: bez różnicowania.
# W logu bez nagłówków kolumna wartości nazywa się "Wartosc".
# Jeżeli w logu występuje kilka nazw liczników, wpisujemy właściwą w "licznik_nazwa".
STACJE = {
    "PZ1": {
        "przeplyw_plik": WEJSCIE / "PZ1_2024.csv",
        "przeplyw_format": "log",       # "log" lub "raport"
        "przeplyw_tryb": "licznik",    # "licznik" lub "godzinowe"
        "przeplyw_kolumna": "Wartosc",
        "licznik_nazwa": None,          # np. "Q1sr_PZ1.2"
        "energia_plik": WEJSCIE / "Energia_PZ1_2024.csv",
        "energia_tryb": "godzinowe",
        "energia_kolumna": None,       # nazwa kolumny kWh z raportu
    },
    "PNB1": {
        "przeplyw_plik": WEJSCIE / "PNB1_2024.csv",
        "przeplyw_format": "log",
        "przeplyw_tryb": "licznik",
        "przeplyw_kolumna": "Wartosc",
        "licznik_nazwa": None,
        "energia_plik": WEJSCIE / "Energia_PNB1_2025_2026.csv",
        "energia_tryb": "godzinowe",
        "energia_kolumna": None,
    },
}

# Daty potwierdzonych błędów pomiaru (jeśli występują w nowym pliku).
# np "PZ1": [("2024-03-05 00:00", "2024-03-20 23:00")]
WYKLUCZONE_OKRESY = {"PZ1": [], "PNB1": []}

STREFA = timezone(timedelta(hours=1))
LOG_KOLUMNY = ["Data", "Godzina", "Licznik", "Wartosc", "Sposob_pomiaru"]

# ====================== PRZETWARZANIE DANYCH =========================


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
        if df.shape[1] != 5:
            raise ValueError(
                f"{plik.name}: log powinien mieć 5 kolumn; ma {df.shape[1]}. "
                "Sprawdź separator i ustaw przeplyw_format='raport', jeśli jest nagłówek."
            )
        df.columns = LOG_KOLUMNY
        # Niektóre pliki mają mimo wszystko w pierwszym wierszu nagłówki.
        df = df.loc[~df["Data"].astype(str).str.strip().eq("Data")].copy()

    return df


def ustaw_czas(df):
    """Polski czas lokalny -> stałe UTC+1; obie problematyczne godziny DST -> NaT."""
    if "Data_czas" in df.columns:
        daty = pd.to_datetime(df["Data_czas"], dayfirst=True, errors="coerce")
    elif "DataCzas" in df.columns:
        daty = pd.to_datetime(df["DataCzas"], dayfirst=True, errors="coerce")
    elif {"Data", "Godzina"}.issubset(df.columns):
        daty = pd.to_datetime(
            df["Data"].astype(str) + " " + df["Godzina"].astype(str),
            dayfirst=True, errors="coerce",
        )
    elif {"Data", "Czas"}.issubset(df.columns):
        daty = pd.to_datetime(
            df["Data"].astype(str) + " " + df["Czas"].astype(str),
            dayfirst=True, errors="coerce",
        )
    else:
        raise ValueError(
            "Nie rozpoznaję kolumn czasu. Dostępne kolumny: " + str(list(df.columns))
        )

    indeks = pd.DatetimeIndex(daty)
    if indeks.tz is None:
        indeks = indeks.tz_localize(
            "Europe/Warsaw", ambiguous="NaT", nonexistent="NaT"
        )
    indeks = indeks.tz_convert(STREFA)
    print("  Odrzucone daty (błędne/DST):", int(indeks.isna().sum()))

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
            pomiary = df["Sposob_pomiaru"].astype(str).str.strip()
            if pomiary.eq("Wartość z pomiaru").any():
                df = df.loc[pomiary.eq("Wartość z pomiaru")].copy()
        print(f"  Wybrany licznik przepływu: {wybor}")

    df = ustaw_czas(df)
    print("  Przykładowe surowe wartości m³:",
          liczby(df, cfg["przeplyw_kolumna"]).dropna().head(5).tolist())
    seria = seria_godzinowa(
        df, cfg["przeplyw_kolumna"], cfg["przeplyw_tryb"], "przepompowano_m3"
    )

    # Odrzucamy wadliwe okresy
    for start, koniec in WYKLUCZONE_OKRESY[stacja]:
        a = pd.Timestamp(start).tz_localize("Europe/Warsaw").tz_convert(STREFA)
        b = pd.Timestamp(koniec).tz_localize("Europe/Warsaw").tz_convert(STREFA)
        seria.loc[(seria.index >= a) & (seria.index <= b + pd.Timedelta(hours=1))] = float("nan")

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

        zapis = WYJSCIE / f"godzinowe-{stacja.lower()}-final.parquet"
        df.to_parquet(zapis)
        print("Zapisano:", zapis)
