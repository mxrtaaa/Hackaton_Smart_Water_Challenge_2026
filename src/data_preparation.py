import pandas as pd
from datetime import timezone, timedelta


def set_timezone(
    df,
    time_col="Data_czas",
    target_tz=timezone(timedelta(hours=1)),
    ambiguous="infer"
):
    """
    Konwertuje znaczniki czasu na stałe UTC+1 (domyślnie).

    Obsługuje:
    - Data_czas jako kolumnę lub indeks,
    - czas bez strefy (interpretowany jako Europe/Warsaw),
    - czas posiadający już strefę,
    - ponowne wywołanie funkcji.

    Zwraca kopię DataFrame z indeksem czasowym.
    """

    result = df.copy()

    # Jeśli czas jest kolumną, ustawiamy go jako indeks.
    if time_col in result.columns:
        result[time_col] = pd.to_datetime(result[time_col])
        result = result.set_index(time_col)

    # Sprawdzamy, czy indeks jest czasowy.
    if not isinstance(result.index, pd.DatetimeIndex):
        raise TypeError(
            f"Brak kolumny '{time_col}' "
            "lub indeks nie jest typu DatetimeIndex."
        )

    # Interpretacja lokalnego czasu polskiego.
    if result.index.tz is None:
        result.index = result.index.tz_localize(
            "Europe/Warsaw",
            ambiguous=ambiguous,
            nonexistent="raise"
        )

    # Konwersja do stałej strefy.
    result.index = result.index.tz_convert(target_tz)

    result.index.name = time_col

    return result

def convert_meter_data(df, kolumna, nowa_nazwa):
    df = df.copy()

    df["Sposób_pomiaru"] = (
        df["Sposób_pomiaru"].astype(str).str.strip()
    )

    df = df[
        df["Sposób_pomiaru"] == "Wartość z pomiaru"
    ].copy()

    df = set_timezone(df)

    # Konwersja wartości liczbowych
    df[kolumna] = pd.to_numeric(
        df[kolumna].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

    licznik = df[kolumna].dropna().sort_index()

    # Duplikaty tego samego momentu
    licznik = licznik.groupby(level=0).max()

    # Ostatni odczyt w każdej godzinie
    hourly = licznik.resample("1h").last()

    # Przyrost godzinowy
    wynik = hourly.diff()

    # Ujemne przyrosty traktujemy jako nieprawidłowe
    wynik = wynik.where(wynik >= 0)

    return wynik.rename(nowa_nazwa)
