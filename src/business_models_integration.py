
import pandas as pd
from sciezka_projektu import get_project_root


ROOT = get_project_root()

BUSINESS = ROOT / "reports" / "chorzow"
MODELS = ROOT / "reports" / "models"

# Wyniki analizy biznesowej Chorzowa
biznes = pd.read_parquet(BUSINESS / "godziny.parquet")
epizody = pd.read_csv(
    BUSINESS / "epizody.csv",
    index_col="epizod_id"
)

# Przypisanie identyfikatora epizodu do mokrych godzin
mokry = biznes["okres"].eq("mokry")
poczatki = mokry & ~mokry.shift(fill_value=False)

biznes["epizod_id"] = (
    poczatki.cumsum().where(mokry).astype("Int64")
)


# Porównujemy oba modele osobno
modele = {
    "CatBoost - przepływ": "chorzow_przeplyw.parquet",
    "LightGBM - energia": "chorzow_energia.parquet",
}

for nazwa, plik in modele.items():

    print(f"\n========== {nazwa} ==========")

    wyniki = pd.read_parquet(MODELS / plik)

    # Ujednolicamy strefę czasową
    if wyniki.index.tz is None:
        raise ValueError(
            f"{nazwa}: indeks wyników nie ma strefy czasowej"
        )

    wyniki.index = wyniki.index.tz_convert(biznes.index.tz)

    if wyniki.index.has_duplicates:
        raise ValueError(
            f"{nazwa}: powtarzające się godziny w wynikach"
        )

    # Zachowujemy godziny z poprawną predykcją
    wyniki = wyniki.dropna(
        subset=["rzeczywiste", "q97"]
    ).copy()

    # Alarm: rzeczywista wartość przekracza Q97
    wyniki["alarm"] = (
        wyniki["rzeczywiste"] > wyniki["q97"]
    )

    # Połączenie godzin modeli i analizy biznesowej
    razem = biznes.join(
        wyniki[["rzeczywiste", "q97", "alarm"]],
        how="inner"
    )

    # Ile alarmów przypadło na każdy mokry epizod?
    ocena = (
        razem.loc[razem["epizod_id"].notna()]
        .groupby("epizod_id")
        .agg(
            godziny_ocenione=("alarm", "size"),
            godziny_alarmowe=("alarm", "sum"),
            wykryty=("alarm", "any")
        )
    )

    porownanie = epizody.join(ocena)

    # Liczymy tylko epizody kompletne:
    # pełne pomiary biznesowe i pełne pokrycie modelem
    pelne = porownanie.loc[
        porownanie["kompletny"].eq(True)
        & porownanie["godziny_ocenione"].eq(
            porownanie["godziny"]
        )
    ].copy()

    wykryte = pelne.loc[pelne["wykryty"].eq(True)]

    print("Godziny ocenione przez model:", len(razem))
    print("Godziny alarmowe:", int(razem["alarm"].sum()))

    print("Pełne epizody ocenione:", len(pelne))
    print("Epizody z przynajmniej jednym alarmem:", len(wykryte))

    if len(pelne) > 0:
        print(
            "Odsetek wykrytych epizodów:",
            round(100 * len(wykryte) / len(pelne), 1),
            "%"
        )

    print(
        "Koszt wszystkich ocenionych epizodów:",
        round(pelne["koszt_zl"].sum(), 2),
        "zł"
    )

    print(
        "Koszt epizodów z alarmem:",
        round(wykryte["koszt_zl"].sum(), 2),
        "zł"
    )

    print("\nNajdroższe epizody z alarmem:")
    print(
        wykryte.sort_values("koszt_zl", ascending=False)[
            ["poczatek", "koniec", "godziny_alarmowe", "koszt_zl"]
        ].head(10)
    )

    # Zapisujemy również porównanie do dalszej analizy
    nazwa_pliku = (
        "catboost_przeplyw" if "CatBoost" in nazwa
        else "lightgbm_energia"
    )

    pelne.to_csv(
        BUSINESS / f"porownanie_{nazwa_pliku}.csv"
    )


#################
# RUDA ŚLĄSKA


import pandas as pd
import numpy as np

from sciezka_projektu import get_project_root


ROOT = get_project_root()
MODELS = ROOT / "reports" / "models"

# Obsługujemy obie lokalizacje raportów biznesowych
BUSINESS = ROOT / "reports" / "ruda-weteranow"

if not (BUSINESS / "godziny.parquet").exists():
    BUSINESS = ROOT / "reports" / "business" / "ruda-weteranow"


# DANE BIZNESOWE WETERANÓW

biznes = pd.read_parquet(
    BUSINESS / "godziny.parquet"
).sort_index()

epizody = pd.read_csv(
    BUSINESS / "epizody.csv",
    index_col="epizod_id"
)

# Przypisujemy każdej mokrej godzinie numer epizodu
mokry = biznes["okres"].eq("mokry")
poczatki = mokry & ~mokry.shift(fill_value=False)

biznes["epizod_id"] = (
    poczatki.cumsum()
    .where(mokry)
    .astype("Int64")
)


# MODELE RUDY ŚLĄSKIEJ — WETERANÓW

modele = [
    (
        "CatBoost - przepływ",
        "ruda_przeplyw.parquet",
        "przepompowano_m3",
        "catboost_przeplyw"
    ),
    (
        "LightGBM - energia",
        "ruda_energia.parquet",
        "energia_kwh",
        "lightgbm_energia"
    )
]


for nazwa, plik, pomiar, nazwa_pliku in modele:

    print(f"\n========== RUDA: {nazwa} ==========")

    wyniki = pd.read_parquet(MODELS / plik).sort_index(kind="stable")

    # Ujednolicamy strefę czasową
    if not isinstance(wyniki.index, pd.DatetimeIndex):
        raise ValueError(f"{nazwa}: brak indeksu czasowego")

    if wyniki.index.tz is None:
        raise ValueError(f"{nazwa}: indeks nie ma strefy czasowej")

    wyniki.index = wyniki.index.tz_convert(
        biznes.index.tz
    )

    if wyniki.index.has_duplicates:
        print(
            f"{nazwa}: usuwam "
            f"{wyniki.index.duplicated().sum()} zduplikowanych godzin"
        )

        wyniki = wyniki.loc[
            ~wyniki.index.duplicated(keep="last")
        ].copy()

    # Tylko godziny z rzeczywistym pomiarem i Q97
    wyniki = wyniki.dropna(
        subset=["rzeczywiste", "q97"]
    ).copy()

    wyniki["alarm"] = (
        wyniki["rzeczywiste"] > wyniki["q97"]
    )

    # KONTROLA ZGODNOŚCI DANYCH
    # Sprawdzamy, czy model i analiza biznesowa
    # korzystają z tych samych pomiarów.

    kontrola = wyniki[["rzeczywiste"]].join(
        biznes[[pomiar]],
        how="inner"
    ).dropna()

    if len(kontrola) > 0:
        zgodne = np.isclose(
            kontrola["rzeczywiste"].to_numpy(),
            kontrola[pomiar].to_numpy(),
            rtol=0.001,
            atol=0.01
        )

        zgodnosc = 100 * zgodne.mean()

        print(
            "Zgodność pomiarów modelu z analizą biznesową:",
            round(zgodnosc, 1),
            "%"
        )

        if zgodnosc < 95:
            print(
                "UWAGA: model i analiza biznesowa korzystają "
                "z różnie przygotowanych pomiarów."
            )

    # ŁĄCZENIE ALARMÓW Z ANALIZĄ BIZNESOWĄ

    razem = biznes.join(
        wyniki[["alarm"]],
        how="inner"
    )

    if razem.empty:
        print("Brak wspólnych godzin.")
        continue

    # Czy podczas epizodu wystąpił przynajmniej jeden alarm?
    ocena = (
        razem.loc[razem["epizod_id"].notna()]
        .groupby("epizod_id")
        .agg(
            godziny_ocenione=("alarm", "size"),
            godziny_alarmowe=("alarm", "sum"),
            wykryty=("alarm", "any")
        )
    )

    porownanie = epizody.join(ocena)

    # Uwzględniamy tylko kompletne epizody,
    # ocenione przez model w każdej godzinie.
    pelne = porownanie.loc[
        porownanie["kompletny"].eq(True)
        & porownanie["godziny_ocenione"].eq(
            porownanie["godziny"]
        )
    ].copy()

    wykryte = pelne.loc[
        pelne["wykryty"].eq(True)
    ]

    koszt_wszystkich = pelne["koszt_zl"].sum()
    koszt_z_alarmem = wykryte["koszt_zl"].sum()

    # PODSUMOWANIE

    print("Godziny ocenione przez model:", len(razem))
    print("Godziny alarmowe:", int(razem["alarm"].sum()))

    print("Pełne epizody ocenione:", len(pelne))
    print("Epizody z przynajmniej jednym alarmem:", len(wykryte))

    if len(pelne) > 0:
        print(
            "Odsetek epizodów z alarmem:",
            round(100 * len(wykryte) / len(pelne), 1),
            "%"
        )

    print(
        "Koszt wszystkich ocenionych epizodów:",
        round(koszt_wszystkich, 2),
        "zł"
    )

    print(
        "Koszt epizodów z alarmem:",
        round(koszt_z_alarmem, 2),
        "zł"
    )

    if koszt_wszystkich > 0:
        print(
            "Udział kosztów epizodów z alarmem:",
            round(100 * koszt_z_alarmem / koszt_wszystkich, 1),
            "%"
        )

    print("\nNajdroższe epizody z alarmem:")

    print(
        wykryte.sort_values(
            "koszt_zl", ascending=False
        )[
            ["poczatek", "koniec", "godziny_alarmowe", "koszt_zl"]
        ].head(10)
    )

    # ZAPIS WYNIKÓW

    output = BUSINESS / f"porownanie_{nazwa_pliku}.csv"

    pelne.to_csv(output)

    print("\nZapisano:", output)
