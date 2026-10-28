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
    start = pd.Timestamp("2026-10-29T00:00:00+01:00").tz_convert("UTC")
    end = start + pd.Timedelta(days=30)
    index = pd.date_range(start-pd.Timedelta(days=7), end, freq="h", inclusive="left")
    frame = pd.DataFrame({"Data_czas": index, "suma_opadu_mm": np.where(np.arange(len(index)) % 80 < 3, .4, 0.)})
    frame.attrs = {"start": start.isoformat(), "endExclusive": end.isoformat(),
                   "model": "ecmwf_ec46", "ensembleMember": 0, "ensembleAggregation": "none"}
    return frame


class RybnikDataTests(unittest.TestCase):
    def fixture(self, root, obj):
        paths = rybnik_processed_paths(root, obj)
        paths["hourly"].parent.mkdir(parents=True)
        times = pd.date_range("2026-09-01", periods=73, freq="h", tz="UTC+01:00")
        keep = times.day != 2
        frame = pd.DataFrame({"przepompowano_m3": 10., "energia_kwh": 2.5, "opad_mm": 999.}, index=times[keep])
        frame.loc[times[0], "energia_kwh"] = 0
        frame.loc[times[5], "energia_kwh"] = np.nan
        frame.loc[times[-1], "energia_kwh"] = np.nan
        frame.to_parquet(paths["hourly"])
        pd.DataFrame({"Data_czas": times, "suma_opadu_mm": np.arange(len(times))*.1}).to_parquet(paths["weather"])
        # Old log/report sources must no longer be read or required.
        for filename in (f"{obj}-2024.parquet", f"raport-{obj}.parquet"):
            (paths["hourly"].parent/filename).write_text("not a parquet")
        return paths

    def test_hourly_sources_preserve_gaps_zero_last_hour_and_external_rain(self):
        for obj in ("pz1", "pnb1"):
            with self.subTest(object=obj), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root, obj)
                frame, info = load_processed_rybnik(root, obj)
                self.assertEqual(frame.index[-1], pd.Timestamp("2026-09-03T23:00:00+01:00"))
                self.assertTrue(frame.loc[frame.index.day == 2, "przepompowano_m3"].isna().all())
                self.assertEqual(frame.loc["2026-09-01 01:00", "przepompowano_m3"], 10)
                self.assertEqual(frame.loc["2026-09-01 00:00", "energia_kwh"], 0)
                self.assertEqual(frame.loc["2026-09-01 03:00", "energia_kwh"], 2.5)
                self.assertTrue(pd.isna(frame.loc["2026-09-01 05:00", "energia_kwh"]))
                self.assertAlmostEqual(frame.opad_mm.iloc[-1], 7.1)
                self.assertIn(f"godzinowe-{obj}-final.parquet", info["files"][0]["file"])
                self.assertFalse(info["excludedLastPartialHour"])
                self.assertEqual(info["lastMeasurements"]["energia_kwh"], frame.index[-1].isoformat())
                self.assertTrue(frame.iloc[-1][["przepompowano_m3", "energia_kwh"]].notna().all())

    def test_missing_hourly_source_cannot_fall_back_to_logs_or_other_station(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.fixture(root, "pz1")
            paths["hourly"].unlink()
            with self.assertRaisesRegex(FileNotFoundError, "godzinowe-pz1-final.parquet"):
                load_processed_rybnik(root, "pz1")
        with self.assertRaises(ValueError):
            rybnik_processed_paths(ROOT, "other")

    def test_empty_and_duplicate_hourly_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.fixture(root, "pz1")
            frame = pd.read_parquet(paths["hourly"])
            pd.concat([frame, frame.iloc[[0]]]).to_parquet(paths["hourly"])
            with self.assertRaisesRegex(ValueError, "unikalny indeks"):
                load_processed_rybnik(root, "pz1")
            frame.iloc[:0].to_parquet(paths["hourly"])
            with self.assertRaisesRegex(ValueError, "Brak godzinowych pomiarów"):
                load_processed_rybnik(root, "pz1")


class RybnikModelTests(unittest.TestCase):
    def test_exact_final_weights_and_real_source_per_station(self):
        for sid, obj in (("rybnik_pz1", "pz1"), ("rybnik_pnb1", "pnb1")):
            with self.subTest(station=sid):
                paths = backend.asset_paths(backend.station_config(sid))
                self.assertEqual(paths[0].name, f"godzinowe-{obj}-final.parquet")
                self.assertTrue(all(p.parent == backend.FINAL_MODELS for p in paths[1:]))
                self.assertTrue(all(f"_{obj}_" in p.name for p in paths[1:]))
                frame, info = load_processed_rybnik(ROOT, obj)
                self.assertIn(f"godzinowe-{obj}-final.parquet", info["files"][0]["file"])
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

    def test_30_day_forecast_starts_october_29_and_uses_predicted_pumping_not_actuals(self):
        for sid in ("rybnik_pz1", "rybnik_pnb1"):
            with self.subTest(station=sid):
                signature = backend.fingerprint(sid)
                backend._rybnik_forecast.cache_clear()
                with patch.object(backend, "load_weather_forecast", return_value=future_weather()) as weather, \
                     patch.object(backend, "energy_features", wraps=backend.energy_features) as features:
                    public, projected, info = backend._rybnik_forecast(sid, signature, "2026-10-29T00:00:00+01:00", "test")
                self.assertEqual(weather.call_args.args, (50.114718, 18.533368))
                self.assertEqual(weather.call_args.kwargs["forecast_days"], 30)
                inputs = features.call_args.args[0]
                pd.testing.assert_series_equal(inputs["przepompowano_m3"].reindex(public.index), public["prognoza"], check_names=False)
                self.assertEqual(len(public), 720)
                self.assertEqual(public.index[0], pd.Timestamp("2026-10-29T00:00:00+01:00"))
                self.assertEqual(public.index[-1], pd.Timestamp("2026-11-27T23:00:00+01:00"))
                pd.testing.assert_series_equal(public.opad_mm, future_weather().set_index("Data_czas").suma_opadu_mm.reindex(public.index), check_names=False)
                pd.testing.assert_series_equal(backend.energy_features(inputs, sid)["przepompowano_m3"].reindex(public.index), public.prognoza, check_names=False)
                self.assertTrue(public[["przepompowano_m3", "energia_kwh"]].isna().all().all())
                self.assertTrue(public["data_kind"].eq("prediction").all())
                self.assertEqual(public["energia_prognoza"].count(), 720)
                self.assertEqual(info["ensembleAggregation"], "none")
                self.assertEqual(info["status"], "ok")
                models = backend.exported_models(sid, signature)
                _, model, names, _ = next(m for m in models if m[0] == "energia_prognoza")
                expected = backend.infer(model, backend.energy_features(inputs, sid), names).reindex(public.index)
                pd.testing.assert_series_equal(public["energia_prognoza"], expected, check_names=False)

    def test_history_requires_joint_measurements_not_just_separate_series_ends(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = RybnikDataTests().fixture(root, "pz1")
            frame = pd.read_parquet(paths["hourly"])
            frame["energia_kwh"] = np.nan
            frame.iloc[-1, frame.columns.get_loc("energia_kwh")] = 2.
            frame.iloc[-1, frame.columns.get_loc("przepompowano_m3")] = np.nan
            frame.to_parquet(paths["hourly"])
            with self.assertRaisesRegex(ValueError, "w tej samej godzinie"):
                load_processed_rybnik(root, "pz1")

    def test_hourly_weather_keeps_selected_member_without_averaging(self):
        source = future_weather()
        values = source.suma_opadu_mm.tolist()
        payload = {"hourly_units": {"precipitation": "mm"}, "hourly": {
            "time": [int(t.timestamp()) for t in source.Data_czas],
            "precipitation": values, "precipitation_member01": [100.] * len(values)}}
        urls = []
        def request(url):
            urls.append(url)
            return payload
        weather = backend.load_weather_forecast(50.114718, 18.533368, forecast_days=30,
            now=pd.Timestamp("2026-10-28T12:00:00Z"), request_json=request)
        self.assertIn("temporal_resolution=hourly", urls[0])
        self.assertIn("models=ecmwf_ec46", urls[0])
        expected = source.set_index("Data_czas").suma_opadu_mm.reindex(weather.Data_czas)
        np.testing.assert_array_equal(weather.suma_opadu_mm, expected)
        self.assertEqual(weather.attrs["ensembleAggregation"], "none")
        with self.assertRaisesRegex(ValueError, "Pusta lub niekompletna"):
            backend.load_weather_forecast(50.114718, 18.533368, forecast_days=1,
                now=pd.Timestamp("2026-10-28T12:00:00Z"), request_json=lambda url: {})

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
