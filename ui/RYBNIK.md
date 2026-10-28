# Rybnik: oddzielne zakładki PZ1 i PNB1

Uruchomienie z katalogu głównego:

```powershell
node ui/server.mjs
```

Adresy na porcie wypisanym przez serwer:

- `/?station=rybnik_pz1`
- `/?station=rybnik_pnb1`

## Historia z aktualnych tabel godzinowych

`src/data_loader.py:load_processed_rybnik` czyta gotowe tabele final z folderu
`data/processed/Rybnik` wskazanego przez użytkownika, bez uruchamiania treningu.
Nie wymaga starych logów liczników ani raportów.

| Obiekt | Plik rzeczywistych m³ i kWh |
| --- | --- |
| PZ1 | `data/processed/Rybnik/godzinowe-pz1-final.parquet` |
| PNB1 | `data/processed/Rybnik/godzinowe-pnb1-final.parquet` |

Wspólne archiwalne opady: `data/processed/Rybnik/suma_opadow.parquet`.
Nazwy z prośby zostały dopasowane do rzeczywistych nazw plików w repozytorium.

Kolumny: `przepompowano_m3`, `energia_kwh`, `opad_mm`; indeks godzinowy ze stałą
strefą UTC+1. m³ i kWh są już przyrostami, nie liczymy ponownie `diff()`.
Opady historyczne pochodzą z `suma_opadow.parquet` (`Data_czas`, `suma_opadu_mm`);
w aktualnych plikach zgadzają się z `opad_mm` tabel final. Łączenie odbywa się po
rzeczywistych znacznikach czasu, bez średnich. Brakujące godziny i ujemne pomiary
m³/kWh pozostają nieznane. Duplikaty i brak strefy czasu powodują czytelny błąd.

Historia 7/14/30 dni jest liczona wstecz od ostatniej dostępnej godziny,
w której są **jednocześnie poprawne pomiary m³ i kWh**. Późniejsze godziny
z samym pompowaniem nie przesuwają końca historii. W plikach z 28.10.2026:

- PZ1: ostatnia wspólna godzina 30.09.2025 23:00 UTC+1, czyli **01.10.2025 00:00 Europe/Warsaw**;
- PNB1: ostatnia wspólna godzina 28.09.2026 12:00 UTC+1, czyli **28.09.2026 13:00 Europe/Warsaw**.

Obie serie w kaflach ostatniej godziny dotyczą tego samego rzeczywistego czasu.
Luki wewnątrz wybranego zakresu nadal pozostają nieznane; nie wybieramy 720
rozproszonych kompletnych wierszy zamiast 30 kolejnych dni. Predykcja nie
zastępuje pomiaru ani w historii, ani w kalkulatorze ustawionym na pomiary.

## Wagi modeli i przyszłość

Wyłącznie `models/final/wyeksportowane modele/`:

- `rybnik_przepompowanie_pz1_median.cbm`, `rybnik_przepompowanie_pz1_q97.cbm`;
- `rybnik_energia_pz1_median.txt`, `rybnik_energia_pz1_q97.txt`;
- `rybnik_przepompowanie_pnb1_median.cbm`, `rybnik_przepompowanie_pnb1_q97.cbm`;
- `rybnik_energia_pnb1_median.txt`, `rybnik_energia_pnb1_q97.txt`.

Nie uruchamiamy treningu ani skryptów `optimized`. CatBoost ma 20 cech
kalendarzowo-opadowych, LightGBM 11 cech, w tym pompowanie. Zachowany jest
kalendarz UTC+1 oraz `min_periods=1` sum/maksimum opadów CatBoost z przygotowania
final. Brakujący bieżący opad i lagi pozostają brakujące; nie podstawiamy zera.
Nowe modele energii wymagają nazwy cechy `przepompowano_m3` (starsze modele
Chorzowa/Rudy zachowują `przepompowano`). Modele energii dostają predykcję pompowania również w porównaniu historycznym;
rzeczywiste kWh są osobną serią.

Prognoza obu obiektów obejmuje **30 dni (720 kolejnych godzin)** od jutrzejszej
północy w Polsce, tak jak w pozostałych zakładkach. Przy uruchomieniu 28.10.2026
jest to **29.10.2026 00:00–27.11.2026 23:00 Europe/Warsaw**. 29 października jest
początkiem, a nie końcem prognozy. Zakres przesuwa się wraz z datą uruchomienia;
nie cofamy aktualnej pogody do dat archiwalnych pomiarów.
UI, zakresy dat kalkulatora i sumy obejmują pełne 30 dni. Opady z
`load_weather_forecast`: EC46, stały scenariusz 0, bez średniej zespołu.
Open-Meteo interpoluje natywny krok 6 h do odpowiedzi 1 h; aplikacja nie wygładza
predykcji. Współrzędne 50.114718, 18.533368 pochodzą z istniejącej konfiguracji
Rybnika w `src/weather_loader.py`. W repozytorium nie znaleziono osobnych punktów
pogodowych dla PZ1 i PNB1, więc oba używają tego punktu i cache
`ui/.runtime/rybnik-weather.json`.

To dane godzinowe odpowiedzi API, nie natywna prognoza pogody w kroku 1 h.
[Dokumentacja Open-Meteo EC46](https://open-meteo.com/en/docs/seasonal-forecast-api).

W przyszłości pompowanie pochodzi wyłącznie z CatBoost, a energia z LightGBM
zasilanego tym pompowaniem i pogodą. Pola pomiarów rzeczywistych są `null`.
Brak pogody pozostawia działającą historię i komunikat o braku prognozy.

## Kalkulator i anomalie

Obie zakładki mają zakresy 7/14/30 dni, Q97 energii, własne daty wyceny,
wybór pomiar/predykcja w historii oraz prognozę kosztów. Przyszłość zawsze
wycenia predykcje. Bilans mokrych okresów i oba koszty anomalii wykorzystują
`src/business_metrics.py`, bez nowych taryf i kosztów napraw.
Demo i `UI_DATA_FILE` nie zastępują źródeł obu obiektów Rybnika.

Sprawdzenie:

```powershell
ui/.runtime/python/python.exe -X utf8 -W ignore -m unittest discover -s ui/tests -p 'test_rybnik.py' -q
node --test ui/tests/rybnik-api.test.mjs
```
