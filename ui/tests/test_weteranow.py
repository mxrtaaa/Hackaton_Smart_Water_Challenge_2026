import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_backend as backend
from data_loader import load_processed_weteranow
import pandas as pd
import numpy as np


class WeteranowTests(unittest.TestCase):
    def test_allowlist_has_only_approved_weteranow_files(self):
        paths = backend.weteranow_paths(backend.ROOT)
        self.assertEqual({p.name for p in paths.values()}, {
            "godzinowe-weteranow.parquet", "scada-weteranow.parquet", "energia-weteranow.parquet",
            "ruda_przepompowanie_median.cbm", "ruda_przepompowanie_q97.cbm", "ruda_energia_median.txt", "ruda_energia_q97.txt"})
        for p in paths.values():
            self.assertNotIn("reports", p.parts)
            self.assertNotIn("chorzow", str(p).lower())
            self.assertNotIn("zmp", p.name)

    def test_missing_approved_files_do_not_fall_back_to_another_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alternative = root / "reports" / "ruda-weteranow"
            alternative.mkdir(parents=True)
            (alternative / "godziny.parquet").write_text("not an approved source")
            with self.assertRaisesRegex(FileNotFoundError, "zatwierdzonych plików Weteranów"):
                backend.require_weteranow(root)

    def test_actual_history_is_preserved_and_last_partial_hour_removed(self):
        frame, metadata = load_processed_weteranow(backend.ROOT)
        source = pd.read_parquet(backend.weteranow_paths(backend.ROOT)["hourly"])
        pd.testing.assert_frame_equal(frame, source.loc[frame.index], check_freq=False)
        self.assertTrue(frame["przepompowano_m3"].isna().any())
        self.assertLess(frame.index[-1], source.index[-1])
        self.assertIn("braków nie interpolujemy", metadata["historyRule"])
        for days in (7, 14, 30):
            self.assertEqual(len(backend.window_detail(frame, days)), 24 * days)

    def test_historical_energy_model_receives_predicted_pumping(self):
        signature = backend.fingerprint("ruda-weteranow")
        detail, _ = backend.load_station("ruda-weteranow", signature)
        raw, _ = load_processed_weteranow(backend.ROOT)
        inputs = raw.copy()
        inputs["przepompowano_m3"] = detail["prognoza"]
        features = backend.energy_features(inputs, "ruda-weteranow")
        models = {column:(model,names) for column,model,names,_ in backend.exported_models("ruda-weteranow", signature)}
        model, names = models["energia_prognoza"]
        expected = backend.infer(model, features, names)
        pd.testing.assert_series_equal(detail["energia_prognoza"], expected, check_names=False)
        self.assertTrue(detail["energia_q97"].notna().any())

    def test_cost_dates_and_predicted_or_measured_energy(self):
        times = pd.date_range("2026-01-01", periods=48, freq="h", tz="Europe/Warsaw")
        detail = pd.DataFrame({"przepompowano_m3":4., "energia_kwh":2., "opad_mm":0.,
                               "prognoza":5., "energia_prognoza":3., "okres":"suchy", "bazowe_m3":4., "bazowe_kwh":2.},index=times)
        prices = {"lowPrice":1, "basePrice":2, "highPrice":3}
        with patch.object(backend, "load_station", return_value=(detail, [])):
            actual = backend.cost_scenarios("ruda-weteranow",14,prices,"history","energy","2026-01-02","2026-01-02")
            predicted = backend.cost_scenarios("ruda-weteranow",14,prices,"history","energy","2026-01-02","2026-01-02","predicted")
        self.assertEqual(actual[1]["valuation"]["energyKwh"],48)
        self.assertEqual(actual[1]["valuation"]["totalCostZl"],96)
        self.assertEqual(actual[1]["valuation"]["pumpingM3"],96)
        self.assertEqual(predicted[1]["valuation"]["totalCostZl"],144)
        self.assertEqual(predicted[1]["valuation"]["dataKind"],"prediction")

    def test_invalid_dates_are_rejected(self):
        index=pd.date_range("2026-01-01",periods=48,freq="h",tz="Europe/Warsaw")
        frame=pd.DataFrame({"energia_kwh":1.},index=index)
        for start,end in (("2026-01-03","2026-01-01"),("2025-12-31","2026-01-01"),("2026-01-01",None),("invalid","2026-01-01")):
            with self.assertRaises(ValueError): backend.select_cost_dates(frame,start,end)

    def test_weather_failure_keeps_only_weteranow_history_available(self):
        with patch.object(backend,"weteranow_forecast",side_effect=ValueError("Weather unavailable")):
            data=backend.dashboard("ruda-weteranow")
        self.assertEqual(data["forecast"],[])
        self.assertEqual(data["weatherForecast"]["status"],"unavailable")
        self.assertEqual(len(data["history"]),720)
        self.assertEqual(data["source"],"weteranow-models")
        self.assertIn("predykcja CatBoost", data["provenance"]["energyMethod"])
        self.assertTrue(all("ruda_" in p["path"] for p in data["provenance"]["models"]))


if __name__ == "__main__": unittest.main()
