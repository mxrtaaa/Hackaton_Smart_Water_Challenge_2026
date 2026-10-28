"""Integration checks against actual saved models and original Python functions."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_backend as backend
import business_metrics
import pandas as pd
from catboost import CatBoostRegressor


class ModelIntegrationTests(unittest.TestCase):
    def test_real_inference_and_original_baseline_are_executed(self):
        backend.load_station.cache_clear()
        with patch.object(business_metrics, "calculate_dry_baseline", wraps=business_metrics.calculate_dry_baseline) as baseline, \
             patch.object(business_metrics, "calculate_excess", wraps=business_metrics.calculate_excess) as excess, \
             patch.object(backend, "run_business_analysis", wraps=backend.run_business_analysis) as analysis:
            with patch.object(backend, "chorzow_forecast", side_effect=ValueError("Offline test")):
                data = backend.dashboard("chorzow")
        self.assertEqual(data["businessEngine"], "python")
        self.assertEqual(data["source"], "processed-models")
        self.assertEqual(data["forecast"], [])
        self.assertTrue(analysis.called and baseline.called and excess.called)
        self.assertEqual(len(data["history"]), 720)
        self.assertTrue(any(r["prognoza"] is not None for r in data["history"]))
        json.dumps(data, allow_nan=False)

    def test_full_history_aggregation_matches_original_run_business_analysis(self):
        station = backend.station_config("chorzow")
        raw = backend.load_processed_chorzow(backend.ROOT)[0]
        original = business_metrics.run_business_analysis(raw)
        actual = backend.summarize(original["detail"], 1)
        self.assertEqual(actual["summary"], backend.clean(original["summary"].to_dict()))
        self.assertEqual(actual["episodes"], backend.records(original["episodes"], "epizod_id"))
        self.assertEqual(actual["monthly"], backend.records(original["monthly"], "miesiac"))

    def test_saved_catboost_is_used_without_retraining_or_scaling_outputs(self):
        station = backend.station_config("chorzow")
        raw = backend.load_processed_chorzow(backend.ROOT)[0]
        model = CatBoostRegressor()
        model.load_model(str(backend.asset_paths(station)[1]))
        features = backend.flow_features(raw).loc[:, model.feature_names_]
        last = features.dropna().iloc[-1:]
        expected = float(model.predict(last)[0])
        detail, _ = backend.load_station("chorzow", backend.fingerprint("chorzow"))
        self.assertAlmostEqual(detail.loc[last.index[0], "prognoza"], expected, places=10)

    def test_missing_measurements_are_preserved_for_both_stations(self):
        for station in backend.STATIONS:
            raw = backend.load_processed_chorzow(backend.ROOT)[0] if station["id"] == "chorzow" else backend.load_processed_weteranow(backend.ROOT)[0]
            detail, _ = backend.load_station(station["id"], backend.fingerprint(station["id"]))
            pd.testing.assert_frame_equal(raw[["przepompowano_m3", "energia_kwh"]], detail[["przepompowano_m3", "energia_kwh"]])
            missing_flow = detail["przepompowano_m3"].isna()
            if station["id"] == "chorzow":
                self.assertTrue(detail.loc[missing_flow, "energia_prognoza"].isna().all())

    def test_price_scenarios_use_original_episode_cost_and_not_hourly_clipping(self):
        detail, _ = backend.load_station("ruda-weteranow", backend.fingerprint("ruda-weteranow"))
        selected = backend.window_detail(detail, 30)
        scenarios = backend.cost_scenarios("ruda-weteranow", 30, {"lowPrice": .5, "basePrice": 1, "highPrice": 2})
        for scenario in scenarios:
            expected = business_metrics.summarize_rain_events(
                business_metrics.calculate_excess(selected, scenario["price"]), scenario["price"])
            self.assertEqual(scenario["episodes"], backend.records(expected, "epizod_id"))
        self.assertAlmostEqual(scenarios[2]["summary"]["koszt_pelnych_epizodow_zl"], 2 * scenarios[1]["summary"]["koszt_pelnych_epizodow_zl"])

    def test_invalid_station_or_prices_do_not_return_demo_or_zero_cost(self):
        with self.assertRaises(ValueError):
            backend.dashboard("missing")
        for prices in ({"lowPrice": -1, "basePrice": 1, "highPrice": 2}, {"lowPrice": 1, "basePrice": None, "highPrice": 2}, {"lowPrice": 2, "basePrice": 1, "highPrice": 3}):
            with self.assertRaises(ValueError):
                backend.cost_scenarios("chorzow", 30, prices)


if __name__ == "__main__":
    unittest.main()
