
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
