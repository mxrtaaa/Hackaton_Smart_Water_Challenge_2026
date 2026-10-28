import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ui"))
import model_backend as backend
from src.data_loader import load_processed_rybnik, rybnik_processed_paths
import pandas as pd
import numpy as np


def future_weather():
    start = pd.Timestamp("2027-01-11T00:00:00+01:00").tz_convert("UTC")
    end = start + pd.Timedelta(days=30)
    index = pd.date_range(start-pd.Timedelta(days=7), end, freq="h", inclusive="left")
    frame = pd.DataFrame({"Data_czas": index, "suma_opadu_mm": np.where(np.arange(len(index)) % 80 < 3, .4, 0.)})
    frame.attrs = {"start": start.isoformat(), "endExclusive": end.isoformat(),
                   "model": "ecmwf_ec46", "ensembleMember": 0, "ensembleAggregation": "none"}
    return frame


class RybnikDataTests(unittest.TestCase):
    def fixture(self, root, obj):
        paths = rybnik_processed_paths(root, obj)
        paths["flow"].parent.mkdir(parents=True)
        times = pd.date_range("2024-12-01", periods=73, freq="h")
        keep = times.day != 2
        meter = "Q1sr_PZ1.2" if obj == "pz1" else "Q1sr_PN1"
        pd.DataFrame({"Data_czas": times[keep], "Licznik": meter, "Sposob_pomiaru": "Wartość z pomiaru ",
                      "Wartosc": 100 + np.arange(len(times))[keep]*10}).to_parquet(paths["flow"])
        energy = pd.DataFrame({"Data_czas": times, "Rodzaj_energii": "energia czynna pobrana",
                               "Wartosc_kWh_kvar": "2,5", "Kompletnosc_procent": 100})
        energy.loc[0, "Wartosc_kWh_kvar"] = "0"
        energy.loc[5, "Kompletnosc_procent"] = 50
        duplicate = energy.iloc[[3]].copy()
        duplicate["Wartosc_kWh_kvar"] = "999"
        pd.concat([energy, duplicate], ignore_index=True).to_parquet(paths["energy"])
        pd.DataFrame({"Data_czas": times, "suma_opadu_mm": 0.}).to_parquet(paths["weather"])
        # This intentionally invalid prepared file must never be read.
        (paths["flow"].parent/f"godzinowe-{obj}-final.parquet").write_text("not a parquet")
        return paths

    def test_raw_logs_and_reports_preserve_gaps_zero_and_complete_hours(self):
        for obj in ("pz1", "pnb1"):
            with self.subTest(object=obj), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root, obj)
                frame, info = load_processed_rybnik(root, obj)
                self.assertEqual(frame.index[-1].tz_convert("Europe/Warsaw"), pd.Timestamp("2024-12-03T23:00:00+01:00"))
                self.assertTrue(frame.loc[frame.index.day == 2, "przepompowano_m3"].isna().all())
                self.assertEqual(frame.loc["2024-12-01 01:00", "przepompowano_m3"], 10)
                self.assertEqual(frame.loc["2024-12-01 00:00", "energia_kwh"], 0)
                self.assertEqual(frame.loc["2024-12-01 03:00", "energia_kwh"], 2.5)
                self.assertTrue(pd.isna(frame.loc["2024-12-01 05:00", "energia_kwh"]))
                self.assertTrue(all("godzinowe" not in p["file"] for p in info["files"]))
                self.assertTrue(info["excludedLastPartialHour"])

    def test_missing_raw_source_cannot_fall_back_to_prepared_or_other_station(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.fixture(root, "pz1")
            paths["flow"].unlink()
            with self.assertRaisesRegex(FileNotFoundError, "pz1-2024.parquet"):
                load_processed_rybnik(root, "pz1")
        with self.assertRaises(ValueError):
            rybnik_processed_paths(ROOT, "other")


class RybnikModelTests(unittest.TestCase):
    def test_exact_final_weights_and_real_source_per_station(self):
        for sid, obj in (("rybnik_pz1", "pz1"), ("rybnik_pnb1", "pnb1")):
            with self.subTest(station=sid):
                paths = backend.asset_paths(backend.station_config(sid))
                self.assertEqual(paths[0].name, f"{obj}-2024.parquet")
                self.assertTrue(all(p.parent == backend.FINAL_MODELS for p in paths[1:]))
                self.assertTrue(all(f"_{obj}_" in p.name for p in paths[1:]))
                frame, info = load_processed_rybnik(ROOT, obj)
                self.assertTrue(all("godzinowe" not in p["file"] for p in info["files"]))
                detail, _ = backend.load_station(sid, backend.fingerprint(sid))
                pd.testing.assert_frame_equal(frame[["przepompowano_m3", "energia_kwh"]],
                                              detail[["przepompowano_m3", "energia_kwh"]])
                for days in (7, 14, 30):
                    self.assertEqual(len(backend.window_detail(detail, days)), days*24)
                for column, model, names, path in backend.exported_models(sid, backend.fingerprint(sid)):
                    if not column.startswith("energia"):
                        features = backend.flow_features(frame, sid)
                        expected = backend.infer(model, features, names)
                        pd.testing.assert_series_equal(detail[column], expected, check_names=False)

    def test_30_day_forecast_uses_predicted_pumping_and_no_future_actuals(self):
        for sid in ("rybnik_pz1", "rybnik_pnb1"):
            with self.subTest(station=sid):
                signature = backend.fingerprint(sid)
                backend._rybnik_forecast.cache_clear()
                with patch.object(backend, "load_weather_forecast", return_value=future_weather()) as weather, \
                     patch.object(backend, "energy_features", wraps=backend.energy_features) as features:
                    public, projected, info = backend._rybnik_forecast(sid, signature, "test", "test")
                self.assertEqual(weather.call_args.args, (50.114718, 18.533368))
                inputs = features.call_args.args[0]
                pd.testing.assert_series_equal(inputs["przepompowano_m3"].reindex(public.index), public["prognoza"], check_names=False)
                self.assertEqual(len(public), 720)
                self.assertTrue(public[["przepompowano_m3", "energia_kwh"]].isna().all().all())
                self.assertTrue(public["data_kind"].eq("prediction").all())
                self.assertEqual(public["energia_prognoza"].count(), 720)
                self.assertEqual(info["ensembleAggregation"], "none")
                self.assertEqual(info["status"], "ok")
                models = backend.exported_models(sid, signature)
                _, model, names, _ = next(m for m in models if m[0] == "energia_prognoza")
                expected = backend.infer(model, backend.energy_features(inputs, sid), names).reindex(public.index)
                pd.testing.assert_series_equal(public["energia_prognoza"], expected, check_names=False)

    def test_both_calculators_select_dates_and_predicted_or_measured_energy(self):
        index = pd.date_range("2026-01-01", periods=48, freq="h", tz="Europe/Warsaw")
        frame = pd.DataFrame({"przepompowano_m3": 4., "energia_kwh": 2., "opad_mm": 0.,
                              "prognoza": 5., "energia_prognoza": 3., "okres": "suchy", "bazowe_m3": 4., "bazowe_kwh": 2.}, index=index)
        prices = {"lowPrice": 1, "basePrice": 2, "highPrice": 3}
        for sid in ("rybnik_pz1", "rybnik_pnb1"):
            with self.subTest(station=sid), patch.object(backend, "load_station", return_value=(frame, [])):
                measured = backend.cost_scenarios(sid, 14, prices, "history", "energy", "2026-01-02", "2026-01-02")
                predicted = backend.cost_scenarios(sid, 14, prices, "history", "energy", "2026-01-02", "2026-01-02", "predicted")
                self.assertEqual(measured[1]["valuation"]["totalCostZl"], 96)
                self.assertEqual(predicted[1]["valuation"]["totalCostZl"], 144)
                self.assertEqual(predicted[1]["valuation"]["pumpingM3"], 120)
                self.assertEqual(predicted[1]["valuation"]["dataKind"], "prediction")
                projected = frame.assign(energia_kwh=3., przepompowano_m3=5.)
                with patch.object(backend, "rybnik_forecast", return_value=(None, projected, {})):
                    future = backend.cost_scenarios(sid, 30, prices, "forecast", "energy", "2026-01-02", "2026-01-02")
                self.assertEqual(future[1]["valuation"]["totalCostZl"], 144)
                self.assertEqual(future[1]["valuation"]["dataKind"], "prediction")

    def test_weather_failure_leaves_both_histories_and_costs_available(self):
        for sid in ("rybnik_pz1", "rybnik_pnb1"):
            with self.subTest(station=sid), patch.object(backend, "rybnik_forecast", side_effect=ValueError("Offline test")):
                data = backend.dashboard(sid)
                self.assertEqual(data["source"], "rybnik-final-models")
                self.assertEqual(data["station"]["id"], sid)
                self.assertEqual(len(data["history"]), 720)
                self.assertEqual(data["forecast"], [])
                self.assertEqual(data["weatherForecast"]["status"], "unavailable")
                self.assertTrue(all(Path(p["path"]).parts[:2] == ("models", "final") for p in data["provenance"]["models"]))

    def test_existing_stations_remain_bound_to_their_own_sources(self):
        for sid, expected in (("chorzow", "processed-models"), ("ruda-weteranow", "weteranow-models")):
            name = "chorzow_forecast" if sid == "chorzow" else "weteranow_forecast"
            with self.subTest(station=sid), patch.object(backend, name, side_effect=ValueError("Offline test")):
                data = backend.dashboard(sid)
                self.assertEqual(data["source"], expected)
                self.assertEqual(len(data["history"]), 720)
                self.assertTrue(all("final" not in p["path"] for p in data["provenance"]["models"]))


if __name__ == "__main__":
    unittest.main()
