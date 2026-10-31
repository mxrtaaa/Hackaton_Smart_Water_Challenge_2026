# .ipynb | Smart Water Challenge 2026

**Team / Zespół:** Marta Załuska · Tomasz Przebindowski · Patrycja Kustra  
**Project:** Wastewater pumping station forecasting, anomaly detection and energy cost analysis

---

## 🇵🇱 Polski

### O projekcie

Projekt **.ipynb**, przygotowany na **Smart Water Challenge 2026**, wspiera analizę pracy przepompowni ścieków. Łączy prognozowanie przepływu i zużycia energii, wykrywanie nietypowych zdarzeń oraz szacowanie kosztów związanych ze zwiększonym dopływem ścieków podczas opadów.

Rozwiązanie rozwijaliśmy na danych z **Chorzowa, Rudy Śląskiej i Rybnika**. Wspólnym punktem odniesienia są godzinowe szeregi pomiarowe, uwzględniające jakość danych, brakujące obserwacje i synchronizację czasu.

### Modele predykcyjne i wykrywanie anomalii

Zastosowaliśmy dwa uzupełniające się podejścia oparte na modelach drzewiastych:

- **CatBoost — prognoza przepływu:** szacuje oczekiwaną objętość przepompowanych ścieków z uwzględnieniem cech czasowych i dostępnych informacji o opadach.
- **LightGBM — prognoza energii:** przewiduje zużycie energii elektrycznej przez przepompownię na podstawie odpowiednich pomiarów i cech opisujących jej pracę.

Dla każdej prognozowanej wielkości wykorzystujemy **model typowej wartości** oraz **model kwantyla Q97**, wyznaczający górną granicę oczekiwanego zachowania. Gdy rzeczywisty odczyt przekracza Q97, system oznacza daną godzinę jako **potencjalną anomalię**. Dzięki temu alarm zależy od warunków pracy i pogody, a nie wyłącznie od stałego progu.

Modele pozwalają również analizować prognozy i odchylenia w czasie. Do interpretacji wpływu cech na predykcje wykorzystujemy **SHAP**. Alarm nie oznacza automatycznie awarii ani nielegalnego przyłącza — jest sygnałem do dalszej weryfikacji.

### Baseline’y i analiza biznesowa

Oprócz prognoz ML przygotowaliśmy **przejrzyste punkty odniesienia (baseline’y)**, które pomagają odpowiedzieć na pytanie, ile przepompownia zużywałaby w typowych warunkach:

- **Baseline suchej pogody:** mediany przepływu i zużycia energii z godzin bez istotnych opadów, z uwzględnieniem miesiąca, godziny i dnia tygodnia/weekendu. Gdy szczegółowa grupa ma zbyt mało obserwacji, stosowany jest prostszy poziom odniesienia.
- **Baseline kalendarzowy:** profil typowej pracy wyznaczany bez korzystania z danych opadowych, przydatny jako alternatywny punkt porównania.
- **Baseline modelowy:** oczekiwane wartości wyznaczane przez modele predykcyjne, również w kontekście warunków pogodowych, jeśli takie dane są dostępne.

Porównujemy pomiary z baseline’ami, obliczając **nadwyżki przepompowanej objętości (m³)** i **energii (kWh)**. Sąsiadujące godziny sklasyfikowane jako mokre łączymy w **epizody opadowe**. Koszt kompletnego epizodu wyznaczamy z dodatniej części sumarycznej nadwyżki energii, przemnożonej przez cenę za kWh. Brakujących pomiarów nie zastępujemy zerami, a niekompletnych epizodów nie przedstawiamy jako w pełni wycenionych.

Następnie zestawiamy alarmy CatBoost i LightGBM z epizodami oraz ich kosztami. Pozwala to wskazywać zdarzenia warte uwagi operatora, szczególnie gdy nietypowe zachowanie występuje podczas kosztownego epizodu.

### Panel operatora i kalkulator budżetu

Interfejs prezentuje **pomiary, prognozy, granice Q97, alarmy, epizody opadowe i oszacowane koszty** dla poszczególnych przepompowni. Ułatwia analizę historii oraz dostępnych prognoz.

W panelu znajduje się również **kalkulator scenariuszy kosztowych (kalkulator budżetu energetycznego)**. Użytkownik może określić zakres analizy i porównać **dolną, bazową oraz górną cenę energii (zł/kWh)**. Kalkulator prezentuje koszt całkowitego zużycia energii albo nadwyżek w mokrych epizodach, korzystając z danych historycznych lub dostępnych prognoz. Zmiana ceny wpływa na wycenę, a nie na prognozowaną ilość ścieków czy energii. Narzędzie służy do analizy wrażliwości kosztów, **nie do obliczania gwarantowanych oszczędności ani zwrotu z inwestycji**.

### Dane i ograniczenia

**Dane pomiarowe udostępnione przez organizatorów nie są publikowane w tym repozytorium.** Kod ilustruje zastosowaną metodykę, ale odtworzenie treningu i wyników wymaga dostępu do odpowiednich danych.

System identyfikuje **odchylenia i zdarzenia wymagające sprawdzenia**; bez potwierdzonych etykiet terenowych nie rozstrzyga przyczyn dopływu. Oszacowane koszty dotyczą przede wszystkim dodatkowego zużycia energii, a nie pełnych kosztów eksploatacji, diagnostyki czy ewentualnych napraw.

---

## 🇬🇧 English

### About the project

**.ipynb** was developed for **Smart Water Challenge 2026** to support the analysis of wastewater pumping station operations. It combines flow and electricity consumption forecasting, anomaly detection, and estimates of the additional energy costs associated with increased wastewater inflow during rainfall.

The project uses data from **Chorzów, Ruda Śląska and Rybnik**. Measurements are aligned into hourly time series, with attention to data quality, missing observations, and time synchronization.

### Predictive models and anomaly detection

We use two complementary tree-based machine learning approaches:

- **CatBoost — flow forecasting:** estimates expected pumped wastewater volumes using temporal features and available rainfall information.
- **LightGBM — energy forecasting:** predicts the electricity consumed by a pumping station using relevant measurements and operational features.

For each target, we train a **typical-value prediction model** and an **upper-quantile model (Q97)**. The Q97 prediction defines an upper bound for expected behavior under the given conditions. An observed value exceeding this bound is flagged as a **potential anomaly**. This provides a context-sensitive alerting mechanism rather than relying solely on fixed thresholds.

The forecasts also support temporal analysis of deviations from expected operation. We use **SHAP** to help explain the influence of input features on predictions. An alert does not establish that a fault or illegal connection exists; it identifies an event worth investigating.

### Baselines and business analysis

In addition to machine learning forecasts, the project includes **interpretable baselines** for assessing normal pumping station operation:

- **Dry-weather baseline:** median flow and electricity usage from sufficiently dry periods, grouped by month, hour, and weekday/weekend characteristics. Less-specific groups are used when there is insufficient data for a detailed reference profile.
- **Calendar baseline:** a typical operating profile constructed without rainfall information, providing an alternative comparison.
- **Model-based baseline:** expected behavior predicted by the machine learning models, accounting for weather conditions where available.

By comparing measurements against these baselines, we estimate **excess pumped volume (m³)** and **excess electricity consumption (kWh)**. Consecutive wet hours are grouped into **rainfall events**. The estimated cost of a complete event is calculated from the positive part of its net excess energy multiplied by the electricity price. Missing readings are not treated as zero, and incomplete events are not reported as fully costed.

We then relate CatBoost and LightGBM alerts to the rainfall events and their estimated costs. This helps operators prioritize unusual behavior, especially when it occurs during potentially expensive events.

### Operator dashboard and budget calculator

The dashboard presents **measurements, forecasts, Q97 thresholds, alerts, rainfall events, and estimated costs** for individual pumping stations. It provides access to historical analysis and available forecasts.

It also includes an **energy budget and cost-scenario calculator**. Users can select an analysis period and compare **low, baseline, and high electricity price assumptions (PLN/kWh)**. The calculator can estimate either total electricity expenditure or excess costs associated with wet-weather events, using historical measurements or available forecasts. Changing the price changes the financial estimate, not the underlying wastewater or energy prediction. This is a tool for **cost sensitivity analysis**, not a calculation of guaranteed savings or investment returns.

### Data availability and limitations

**Measurement datasets supplied by the competition organizers are not included in this public repository.** The code documents our approach, but reproducing model training and results requires access to the relevant datasets.

The system detects **unusual patterns that warrant further inspection**. Without verified field labels, it cannot determine the precise cause of an anomalous inflow. Reported financial figures primarily estimate additional electricity costs; they do not represent total operating, investigation, or repair costs.
