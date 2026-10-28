# Rybnik: oddzielne zakładki PZ1 i PNB1

Uruchomienie z katalogu głównego:

```powershell
node ui/server.mjs
```

Adresy na porcie wypisanym przez serwer:

- `/?station=rybnik_pz1`
- `/?station=rybnik_pnb1`

## Historia z logów i raportów

`src/data_loader.py:load_processed_rybnik` korzysta z istniejących funkcji
`ustaw_czas` i `seria_godzinowa` z `src/wczytaj_rybnik_final.py`.
Nie czyta gotowych plików `godzinowe-*-final.parquet`.

| Obiekt | Pompowanie: log licznika narastającego | Energia: raport godzinowy |
| --- | --- | --- |
| PZ1 | `data/processed/Rybnik/pz1-2024.parquet`, licznik `Q1sr_PZ1.2` | `data/processed/Rybnik/raport-pz1.parquet` |
| PNB1 | `data/processed/Rybnik/pnb1-2024.parquet`, licznik `Q1sr_PN1` | `data/processed/Rybnik/raport-pnb1.parquet` |

Wspólne archiwalne opady: `data/processed/Rybnik/suma_opadow.parquet`.
Nazwy z prośby zostały dopasowane do rzeczywistych nazw plików w repozytorium.

Pompowanie: tylko „Wartość z pomiaru”, ostatni licznik na godzinę i różnica;
ujemne przyrosty i brakujące godziny pozostają nieznane. Powtórzone znaczniki
czasu zachowują pierwszy odczyt, jak w przygotowaniu modeli final.
Odcięcie poprawnego okresu: 19.03.2024 13:00 UTC+1, zgodnie z kodem treningu final.
Ostatni niepełny przedział logu jest odrzucany.

Energia: wyłącznie energia czynna pobrana, kompletność 100%; wartości są już
godzinowymi kWh, nie są różnicowane. W godzinie zachowany jest pierwszy poprawny
odczyt; nie sumujemy zdublowanych raportów. Niejednoznaczne daty DST pozostają
nieznane. Dane nie są interpolowane ani uzupełniane zerem.

Historia 7/14/30 dni jest liczona wstecz od ostatniej dostępnej pełnej godziny
wspólnego zakresu m³ i kWh. W aktualnych plikach:

- PZ1: 30.12.2024 23:00 Europe/Warsaw;
- PNB1: 31.12.2024 22:00 Europe/Warsaw.

Raport PZ1 sięga 2025 r., ale log pompowania kończy się w 2024 r.; brakującego
pompowania nie dopisujemy. Kafle ostatniej godziny pokazują datę archiwum.

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
Modele energii dostają predykcję pompowania również w porównaniu historycznym;
rzeczywiste kWh są osobną serią.

Prognoza: 720 kolejnych godzin od jutrzejszej północy w Polsce. Opady z
`load_weather_forecast`: EC46, stały scenariusz 0, bez średniej zespołu.
Open-Meteo interpoluje natywny krok 6 h do odpowiedzi 1 h; aplikacja nie wygładza
predykcji. Współrzędne 50.114718, 18.533368 pochodzą z istniejącej konfiguracji
Rybnika w `src/weather_loader.py`. W repozytorium nie znaleziono osobnych punktów
pogodowych dla PZ1 i PNB1, więc oba używają tego punktu i cache
`ui/.runtime/rybnik-weather.json`.

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
