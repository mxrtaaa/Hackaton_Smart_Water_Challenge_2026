# Chorzów — historia pomiarów i prognoza 30 dni

Dodano osobną zakładkę **Ruda Śląska · Weteranów**. Obie lokalizacje korzystają
z funkcji w `src/data_loader.py`; Chorzów zachowuje dotychczasowe modele i źródła.

## Uruchomienie i sprawdzenie

Z katalogu głównego repozytorium, po zatrzymaniu poprzedniego serwera przez Ctrl+C:

```powershell
node ui/server.mjs
```

Otwórz adres wypisany w terminalu (domyślnie http://127.0.0.1:4173, ewentualnie
kolejny wolny port do 4183). Pierwsze wczytanie może zająć kilka sekund; bez cache
pogody wymaga internetu. Lokalne środowisko `ui/.runtime/python` jest już gotowe.
Na innym komputerze ustaw `UI_PYTHON` na Pythona z bibliotekami z
`ui/requirements-models.txt` (np. po utworzeniu osobnego venv i `pip install -r`).
Zastana `.venv` ma biblioteki, ale wskazuje nieistniejący interpreter; przenośny
Python w `ui/.runtime` korzysta z jej bibliotek bez modyfikacji tego środowiska.

```powershell
node --test ui/tests/*.test.mjs
ui/.runtime/python/python.exe -X utf8 -m unittest discover -s ui/tests -p 'test_*.py' -q
```

Domyślnie nie ma symulacji: źródłem są pliki Chorzowa, zapisane modele i Open-Meteo.
Stare demo pozostaje dostępne tylko jawnie przez `UI_SOURCE=demo`; nie jest
fallbackiem błędu danych lub modelu. `UI_DATA_FILE` zachowuje obsługę snapshotów,
ale pełna wycena nowych opcji wymaga połączonego backendu Python.

## Sprawdzone dane Chorzowa

| Plik `data/processed/Chorzów/` | Kolumna | Liczba wierszy | Charakter |
| --- | --- | ---: | --- |
| `scieki.parquet` | `Ilosc_przepompowanych_sciekow` | 414 414 | Nieregularne odczyty narastającego licznika m³ |
| `energia.parquet` | `Zuzycie_energii_elektrycznej` | 664 021 | Nieregularne odczyty narastającego licznika kWh |
| `suma_opadow.parquet` | `suma_opadu_mm` | 24 672 | Historyczne opady godzinowe, z duplikatami/lukami |

Liczniki mają `Data_czas`, `Sposób_pomiaru` i odczyty od 13.10.2024 00:00 do
13.10.2026 00:00 polskiego czasu. Nie należy traktować ich jako godzinowego zużycia.
Adapter wywołuje oryginalne `convert_meter_data`: filtruje „Wartość z pomiaru”,
porządkuje duplikaty, bierze ostatni licznik na godzinę, a następnie różnicuje.
Zera są poprawnymi przyrostami; cofnięcia i braki pozostają nieznane.

Ostatni niepełny przedział resamplingu jest pomijany. W efekcie ostatnia pełna
historyczna godzina ma etykietę **12.10.2026, 23:00 Europe/Warsaw**
(`2026-10-12T22:00:00+01:00` w stałej strefie projektu). Nie sugerujemy, że to
ostatnia godzina względem dzisiejszej daty.

Historia 7 / 14 / 30 dni jest liczona **wstecz od ostatniej wspólnej pełnej godziny
liczników**, nie od dziś. Dostępne są pełne okna 168 / 336 / 720 godzin; brakujące
pomiary wewnątrz okna pozostają `null`. Zakres 14 dni jest dodatkową opcją, a nie
interpolacją brakujących siedmiu dni. Nagłówek i opisy podają datę końca archiwum.
Pierwsze dwa kafle pokazują rzeczywiste m³ i kWh z ostatniej pełnej godziny
archiwum, z datą pomiaru. Nie zmieniają się przy wyborze zakresu wykresu.
Pozostałe dwa pokazują osobno sumę prognozowanych m³ i kWh przyszłych 30 dni
oraz pokrycie predykcji. Brak ostatniego pomiaru nie jest zastępowany starszym.

## Pogoda na kolejne 30 dni

Dotychczasowe `src/weather_loader.py:load_weather_data` i jego archiwalny kontrakt
zostały zachowane. Nowa funkcja `load_weather_forecast` korzysta z:

- endpointu `https://seasonal-api.open-meteo.com/v1/seasonal`,
- modelu `ecmwf_ec46`, pojedynczego scenariusza nr 0 (pole `precipitation`),
- `hourly=precipitation`, `temporal_resolution=hourly`, `timeformat=unixtime`, UTC,
- współrzędnych Chorzowa 50.292298, 18.954600, takich jak w istniejącym loaderze,
- 8 dni kontekstu z API na potrzeby lagów oraz 31 dni odpowiedzi (wybieramy następne
  720 godzin od początku jutrzejszego dnia w Europe/Warsaw).

Zwykły krótki forecast nie zapewnia całego miesiąca. EC46 ma horyzont 46 dni,
więc pokrywa wymagane 30 dni. Natywna rozdzielczość to **6 h**; Open-Meteo
interpoluje ją do 1 h. Używamy jednego stałego scenariusza, bez średniej między
członkami zespołu. To jeden możliwy obszarowy przebieg pogody, bez gwarancji
lokalnej trafności ani wskazania, że jest najbardziej prawdopodobny.
Interpolacja nie zwiększa rozdzielczości informacji.
Nie tworzymy własnych opadów ani nie przepisujemy archiwum jako przyszłej pogody.
Źródło, numer scenariusza (`ensembleMember`), brak agregacji (`ensembleAggregation=none`),
data pobrania, rozdzielczość i pokrycie są w `weatherForecast` i opisie UI.
Scenariusz jest wybierany z góry, a nie ze względu na wygląd wyników.
Parametr `ensemble_member` loadera pozwala jawnie wskazać członka 0–50; brak
wybranego scenariusza powoduje błąd, bez przejścia na średnią lub innego członka.

Źródło: [dokumentacja Seasonal/EC46 Open-Meteo](https://open-meteo.com/en/docs/seasonal-forecast-api).
Cache w `ui/.runtime/chorzow-weather.json` ma TTL 6 h i jest powiązany z początkiem
horyzontu, lokalizacją, modelem i scenariuszem. Poprzedni cache średniej zespołu
nie pasuje do nowego klucza i nie jest ponownie wykorzystywany.
Puste odpowiedzi, inne jednostki i błędna oś
czasu są odrzucane. Brakujące godziny/dni nie są wypełniane zerem. Błąd pogody nie
blokuje historii: UI pokaże brak prognozy i przyczynę, zamiast wyświetlać pomiary
jako predykcje. Cache nie jest publikowany ani śledzony przez Git.

## Pompowanie i energia w przyszłości

CatBoost median / Q97 i LightGBM median / Q97 są ładowane z istniejących folderów
`wyeksportowane modele`, bez treningu i bez modyfikacji plików. Kolejność cech
pochodzi z zapisanych modeli. Odtworzono kalendarz i opóźnienia opadów z kodu
przygotowania danych. Kalendarz CatBoost to stałe UTC+1, energia Chorzowa używa
Europe/Warsaw — zachowano konwencje treningu.

Przyszłe pompowanie pochodzi wyłącznie z CatBoost i cech pogody. LightGBM wymaga
`przepompowano` jako cechy: w przyszłości dostaje **predykcję CatBoost**, nigdy
pomiar ścieków. Pogodowy warm-up służy tylko cechom, nie wypełnia historii pomiarów.
Brak wymaganych cech oznacza brak predykcji, a nie domyślne zero.

Historia kończy się w archiwum, a prognoza zaczyna od jutra — są to odrębne osie
czasu. Przy danych z tego workspace i uruchomieniu 27.10.2026 prognoza obejmuje
**28.10–26.11.2026**, 720 kolejnych godzin. Nie jest to retrospekcja od 13.10.
Horyzont jest czasem rzeczywistym (720 h); przy zmianie czasu granice kalendarzowe
mogą różnić się o godzinę. Q97 modelu nie jest pełnym przedziałem niepewności
pogody i modelu łącznie. Porównania historyczne nie stanowią niezależnej walidacji.
Sprawdzono poprawność przepływu danych i obliczeń, ale nie zmierzono błędu
30-dniowej prognozy na niezależnych historycznych prognozach pogody.
Usunięto poprzednie uśrednianie scenariuszy pogodowych. Pojedynczy scenariusz
również może się nie sprawdzić; podstawienie prognozowanego pompowania do modelu
energii propaguje błąd.
Regularny profil dobowy wykresu może wynikać z cech kalendarzowych i podobnych
warunków pogodowych. Sam kształt wykresu nie potwierdza jakości prognostycznej.

Kontrola **poprzedniego wariantu ze średnią** z 27.10.2026 wykazała, że przyszłe opady są przez dużą część horyzontu
niemal stałe (około 0,1 mm/h). Mediana 720 godzinowych prognoz bazowych wynosiła
26,6 m³/h, a mediana ich progów Q97 — 97,4 m³/h. W kontrolnym przeliczeniu na
tych samych datach, ale przy zerowych opadach, wartości wyniosły odpowiednio
23,2 i 37,9 m³/h. To test wrażliwości, nie nowa prognoza: potwierdza dużą reakcję
modelu Q97 na uśredniony opad. Produkcyjnych wyników nie zastępowano tym testem.
Q97 jest kwantylem warunkowym modelu, nie pełnym przedziałem niepewności prognozy
30-dniowej. Na wykresie rozdzielono jednostkę i nagłówek prognozy, usunięto
rozciąganie tekstu SVG oraz ukryto legendę przekroczeń w widoku przyszłości.

Każdy wiersz `forecast` ma `data_kind='prediction'`, `przepompowano_m3=null` oraz
`energia_kwh=null`. Wyniki są wyłącznie w `prognoza`, `q97`, `energia_prognoza`,
`energia_q97`. Frontend odrzuca wiersz prognozy zawierający pomiary. Przyszłe
nadwyżki mają prefiks `predicted_`; nie są mieszane z historycznym bilansem.

## Kalkulator i metryki biznesowe

Opcje kalkulatora:

1. Okres: jak na wykresie, historia 7/14/30 dni lub przyszłe 30 dni.
2. Podstawa: całe zużycie energii albo nadwyżki w mokrych epizodach.
3. Cena dolna / bazowa / górna, edytowalne; domyślnie cena źródłowa projektu.
4. Dla obu lokalizacji: własne daty od/do w wybranym okresie oraz wybór
   rzeczywistych pomiarów lub predykcji energii w historii. Przyszłość zawsze
   korzysta z predykcji. Wynik zawiera sumy kWh, m³ i pokrycie godzinowe.

Dashboard Chorzowa i Weteranów korzysta ze wspólnego układu, wykresów energii
z Q97 oraz tych samych opcji kalkulatora. Modele, źródła i daty końca archiwum
pozostają właściwe dla lokalizacji. W historii Chorzowa model energii nadal
korzysta ze zmierzonego pompowania (zgodnie z treningiem); w historii Weteranów
z przewidywanego pompowania. W przyszłości oba używają predykcji pompowania.

Całe zużycie to suma rzeczywistych lub przewidywanych kWh × cena; wynik i tabela
dzienna są jawnie oznaczone POMIAR lub PREDYKCJA. Niepełne dane dają wycenę
częściową z liczbą godzin. Nigdy nie zastępujemy brakującej przyszłości historią.
Zakres wariantów wynika z ceny kWh, nie jest przedziałem ufności ani oszczędnością.

Historyczny bilans mokrych okresów nadal wywołuje oryginalny `run_business_analysis`
(z `calculate_dry_baseline` i `calculate_excess`) oraz `summarize_rain_events`.
Klasyfikacja mokro/sucho, zachowanie znaku nadwyżek, obcięcie kosztu po sumie
całego epizodu i niewycenianie niekompletnych epizodów pozostają bez zmian.

Wycena **przewidywanej** mokrej nadwyżki stosuje te same wzory, ale na osobnej,
wewnętrznej ramce predykcji. Baseline powstaje wyłącznie z historycznych suchych
pomiarów. Predykcje nie uczą baseline'u i nie stają się pomiarami w publicznym API.
Mokro/sucho w przyszłości wynika z prognozowanych opadów. Bilans historyczny pod
wykresami nie jest podmieniany przy wyborze przyszłości w kalkulatorze.
Brak kosztów diagnostyki/napraw, ROI i deklarowanych oszczędności.

## Zmiany plików i API

- `src/weather_loader.py`: dodana prognoza; dotychczasowy loader archiwum bez zmian.
- `src/data_loader.py`: wspólne miejsce wczytywania danych Chorzowa i Weteranów;
  `load_processed_chorzow` używa istniejącego `convert_meter_data`, a
  `load_processed_weteranow` czyta przygotowane przyrosty bez ponownego różnicowania.
- `ui/model_backend.py`: inferencja przyszłości, oznaczenia źródeł, wycena i metadane.
- `ui/real-data-source.mjs`, `ui/server.mjs`: nowe opcjonalne parametry wyceny.
- `ui/public/app.mjs`, `index.html`, `view-data.mjs`, `styles.css`: wybór okresu i
  podstawy wyceny, kafle z sumami, kontrola rozdziału danych i usunięcie szczegółów Q97.
- Testy i `ui/requirements-models.txt`: zależności istniejącego loadera pogody.

`GET /api/dashboard?station=chorzow` zachowuje `schemaVersion:2`, `history`,
`forecast`, `businessByRange` i rozszerza odpowiedź o `weatherForecast`,
`historyTotals`, `forecastTotals`, `historyRanges` i pochodzenie danych.

`POST /api/cost-scenarios` zachowuje stare wywołania (`period=history`,
`metric=wet_excess` domyślnie). Nowy przykład:

```json
{"station":"chorzow","days":30,"period":"forecast","metric":"energy",
 "prices":{"lowPrice":0.8,"basePrice":1,"highPrice":1.2}}
```

Odpowiedź zachowuje metryki i dodaje `valuation` z `dataKind`, `totalCostZl`,
`energyKwh`, `pricedHours`, `expectedHours`, `complete`, `daily`. Metadane dzienne
energii są informacyjne przy wyborze mokrych nadwyżek; rozbicie kosztu w UI jest
pokazywane tylko dla całego zużycia, bo epizod może przecinać granicę dni.

Ruda Śląska — Weteranów ma teraz oddzielną zakładkę i zatwierdzoną listę plików
z `data/processed/Ruda Śląska` oraz `models`; nie korzysta z raportów jako fallbacku.
`business_metrics.py`, dane i modele pozostają nietknięte.

Testy obejmują rzeczywiste pliki i tymczasowe liczniki z brakującym dniem,
puste odpowiedzi pogodowe, cache, brak zamiany null na zero, zakaz pomiarów
w prognozie, użycie predykcji pompowania przez model energii, przyszłe ceny,
API oraz zgodność historycznych wzorów z oryginalnym modułem. Testy jednostkowe
pogody używają kontrolowanych odpowiedzi; pełną odpowiedź EC46 oraz inferencję
720 godzin zweryfikowano również na rzeczywistym połączeniu. Kontrola wizualna
pozostaje ograniczona dostępnością narzędzia przeglądarkowego.

## Które pliki są potrzebne (porządkowanie 28.10.2026)

Usunięto `ui/chorzow_data.py` i `ui/weteranow_data.py`. Ich zadania wykonują teraz
funkcje w istniejącym `src/data_loader.py`, używane bezpośrednio przez backend.
Nie zmieniono kontraktu funkcji wczytujących surowe CSV. `load_scada_weterani`
pozostaje funkcją do dotychczasowego przygotowania danych modelowych; interpoluje
braki, więc nie zastępuje źródła historii pomiarów. Nie importujemy
`src/business_analysis.py` w serwerze, bo ten skrypt wykonuje analizę i zapisuje
pliki także przy imporcie. Backend nie korzysta już z nieużywanej funkcji
`load_hourly` odczytującej raporty.

| Pliki / katalog | Rola i możliwość usunięcia |
| --- | --- |
| `server.mjs`, `data-source.mjs`, `real-data-source.mjs`, `python-bridge.mjs`, `model_backend.py` | Potrzebne w obecnej strukturze: serwer, wybór źródła, API i proces Pythona z inferencją. |
| `public/index.html`, `styles.css`, `app.mjs`, `view-data.mjs` | Potrzebne do interfejsu i poprawnego wyboru historii/prognozy. |
| `public/economics.mjs` | Waliduje ceny w interfejsie; dodatkowo wycenia jawne demo/snapshot. Nie usuwać samodzielnie. |
| `public/business-metrics.mjs` | Używany przez demo, wycenę offline i frontend jako wariant zastępczy dla snapshotu. Rzeczywiste dane liczy `src/business_metrics.py`. Nie usuwać bez usunięcia tych ścieżek i importów. |
| `demo-source.mjs` | Opcjonalne demo (`UI_SOURCE=demo`), ale moduł jest importowany przez `data-source.mjs`. Samo skasowanie pliku przerwie start. Pozostawiony dla zachowania dotychczasowej funkcji. |
| `tests/` | Nie są potrzebne do serwowania strony, lecz służą sprawdzeniu modeli, danych, kosztów i API. Warto zachować. |
| `requirements-models.txt` | Zależności Pythona na potrzeby instalacji i odtworzenia środowiska. Zachować. |
| `package.json` | Komendy `npm start` / `npm test` i informacje o Node. Zachować. |
| `README.md`, `.gitignore` | Instrukcja i wykluczenie plików tymczasowych ze śledzenia. Zachować. |
| `__pycache__/`, `*.pyc` | Można usuwać po zatrzymaniu aplikacji; Python odtworzy je w razie potrzeby. |
| `.runtime/chorzow-weather.json`, `.runtime/weteranow-weather.json` | Odtwarzalny cache pogody; po usunięciu ponowne pobranie wymaga internetu. |
| `.runtime/python/` | Obecnie potrzebny interpreter. Usuwać dopiero po ustawieniu `UI_PYTHON` na sprawne środowisko z wymaganymi bibliotekami. |
| `.runtime/pdf-packages/` | Biblioteki do przygotowania wcześniejszego PDF, nie są wymagane do działania panelu. Można usunąć, jeśli dokument nie będzie generowany ponownie. |
| `.venv-models/` (jeśli istnieje) | Nie jest domyślnym interpreterem. Można usunąć wyłącznie gdy żaden `UI_PYTHON` ani inne zadanie nie korzysta z tego środowiska. |

Zależności te sprawdzono dla kodu panelu. Modele, dane, notebooki i moduły innych
zadań projektu nie są usuwane na podstawie tego, że nie korzysta z nich UI.

Weteranów czyta wyłącznie `godzinowe-weteranow.parquet`, `scada-weteranow.parquet`
i `energia-weteranow.parquet` z `data/processed/Ruda Śląska`, a backend wymaga
zatwierdzonych modeli `ruda_przepompowanie_{median,q97}.cbm` oraz
`ruda_energia_{median,q97}.txt`. Dane i modele innych lokalizacji nie zastępują
brakujących plików. Energia Weteranów jest przewidywana z predykcji pompowania
także w historii. Kalkulator zachowuje wybór dat i podstawy: pomiar albo predykcja.
