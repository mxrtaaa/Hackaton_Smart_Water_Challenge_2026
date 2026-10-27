"""Zestawienie alarmów modeli z kosztami epizodów opadowych."""

import numpy as np
import pandas as pd

from sciezka_projektu import get_project_root


ROOT = get_project_root()
MODELS = ROOT / "reports" / "models"

# Obiekt: (katalog raportu biznesowego, modele)
OBIEKTY = {
    "CHORZÓW": (
        "chorzow",
        [
            ("CatBoost - przepływ", "chorzow_przeplyw.parquet", "przepompowano_m3", "catboost_przeplyw"),
            ("LightGBM - energia", "chorzow_energia.parquet", "energia_kwh", "lightgbm_energia"),
        ],
    ),
    "RUDA - WETERANÓW": (
        "ruda-weteranow",
        [
            ("CatBoost - przepływ", "ruda_przeplyw.parquet", "przepompowano_m3", "catboost_przeplyw"),
            ("LightGBM - energia", "ruda_energia.parquet", "energia_kwh", "lightgbm_energia"),
        ],
    ),
}

for obiekt, (katalog, modele) in OBIEKTY.items():
    business_path = ROOT / "reports" / katalog
    biznes = pd.read_parquet(business_path / "godziny.parquet").sort_index()
    epizody = pd.read_csv(business_path / "epizody.csv", index_col="epizod_id")

    if not isinstance(biznes.index, pd.DatetimeIndex) or biznes.index.tz is None:
        raise ValueError(f"{obiekt}: niepoprawny indeks czasowy danych biznesowych")
    if biznes.index.has_duplicates:
        raise ValueError(f"{obiekt}: zduplikowane godziny w danych biznesowych")

    # Numeracja epizodów jest taka sama jak w business_metrics.summarize_rain_events.
    mokry = biznes["okres"].eq("mokry")
    poczatki = mokry & ~mokry.shift(fill_value=False)
    biznes["epizod_id"] = poczatki.cumsum().where(mokry).astype("Int64")

    for nazwa, plik, pomiar, etykieta in modele:
        print(f"\n========== {obiekt}: {nazwa} ==========")

        wyniki = pd.read_parquet(MODELS / plik).sort_index(kind="stable")
        if not isinstance(wyniki.index, pd.DatetimeIndex) or wyniki.index.tz is None:
            raise ValueError(f"{nazwa}: wyniki modelu nie mają indeksu ze strefą czasową")

        wyniki.index = wyniki.index.tz_convert(biznes.index.tz)
        duplikaty = int(wyniki.index.duplicated().sum())
        if duplikaty:
            print(f"Pomijam {duplikaty} powtórzonych godzin (zostawiam ostatni odczyt)")
            wyniki = wyniki.loc[~wyniki.index.duplicated(keep="last")].copy()

        wyniki = wyniki.dropna(subset=["rzeczywiste", "q97"]).copy()
        wyniki["alarm"] = wyniki["rzeczywiste"] > wyniki["q97"]

        # Sprawdzamy zgodność na wspólnych godzinach z dwoma poprawnymi pomiarami.
        kontrola = wyniki[["rzeczywiste"]].join(biznes[[pomiar]], how="inner").dropna()
        if kontrola.empty:
            raise ValueError(f"{obiekt}, {nazwa}: brak wspólnych poprawnych pomiarów")
        zgodne = np.isclose(
            kontrola["rzeczywiste"], kontrola[pomiar], rtol=0.001, atol=0.01
        )
        print(f"Zgodność pomiarów: {100 * zgodne.mean():.1f}% "
              f"({int(zgodne.sum())}/{len(kontrola)} wspólnych poprawnych godzin)")
        if zgodne.mean() < 0.95:
            raise ValueError(f"{obiekt}, {nazwa}: niezgodne pomiary modelu i analizy biznesowej")

        razem = biznes.join(wyniki[["alarm"]], how="inner")
        if razem.empty:
            raise ValueError(f"{obiekt}, {nazwa}: brak wspólnych godzin")

        ocena = (
            razem.loc[razem["epizod_id"].notna()]
            .groupby("epizod_id")
            .agg(
                godziny_ocenione=("alarm", "size"),
                godziny_alarmowe=("alarm", "sum"),
                wykryty=("alarm", "any"),
            )
        )
        porownanie = epizody.join(ocena)
        pelne = porownanie.loc[
            porownanie["kompletny"].eq(True)
            & porownanie["godziny_ocenione"].eq(porownanie["godziny"])
        ].copy()
        wykryte = pelne.loc[pelne["wykryty"].eq(True)]

        koszt = pelne["koszt_zl"].sum()
        koszt_z_alarmem = wykryte["koszt_zl"].sum()

        print(f"Godziny ocenione: {len(razem)}, alarmowe: {int(razem['alarm'].sum())}")
        print(f"Pełne epizody: {len(pelne)}, z alarmem: {len(wykryte)}")
        if len(pelne):
            print(f"Odsetek epizodów z alarmem: {100 * len(wykryte) / len(pelne):.1f}%")
        print(f"Koszt wszystkich epizodów: {koszt:.2f} zł")
        print(f"Koszt epizodów z alarmem: {koszt_z_alarmem:.2f} zł")
        if koszt > 0:
            print(f"Udział kosztów epizodów z alarmem: {100 * koszt_z_alarmem / koszt:.1f}%")

        print("\nNajdroższe epizody z alarmem:")
        print(
            wykryte.sort_values("koszt_zl", ascending=False)[
                ["poczatek", "koniec", "godziny_alarmowe", "koszt_zl"]
            ].head(10)
        )
        output = business_path / f"porownanie_{etykieta}.csv"
        pelne.to_csv(output)
        print("Zapisano:", output)
