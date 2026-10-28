import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_backend as backend
from data_loader import load_processed_chorzow
from weather_loader import load_weather_forecast, forecast_start
import numpy as np
import pandas as pd

NOW = pd.Timestamp("2027-01-10T12:00:00Z")


def weather_payload(missing_day=False):
    start = forecast_start(NOW).tz_convert("UTC")
    times = pd.date_range(start-pd.Timedelta(days=7), start+pd.Timedelta(days=30), freq="h", inclusive="left")
    if missing_day:
        times = times[~((times >= start+pd.Timedelta(days=4)) & (times < start+pd.Timedelta(days=5)))]
    return {"hourly": {"time": [int(t.timestamp()) for t in times], "precipitation": [0.0]*len(times)},
            "hourly_units": {"precipitation": "mm"}}


class ConstantModel:
    def __init__(self, value): self.value = value
    def predict(self, frame): return np.full(len(frame), self.value)


class EnergyModel:
    def predict(self, frame): return frame["przepompowano"].to_numpy()*3


class ChorzowForecastTests(unittest.TestCase):
    def test_processed_loader_reads_counters_and_keeps_missing_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root/"data"/"processed"/"chorzów"
            folder.mkdir(parents=True)
            times=pd.date_range("2026-01-01",periods=73,freq="h")
            keep=times.day!=2
            for file,column in (("scieki.parquet","Ilosc_przepompowanych_sciekow"),("energia.parquet","Zuzycie_energii_elektrycznej")):
                pd.DataFrame({"Data_czas":times[keep],column:np.arange(73)[keep]*10,"Sposób_pomiaru":"Wartość z pomiaru"}).to_parquet(folder/file)
            pd.DataFrame({"Data_czas":times,"suma_opadu_mm":0.0}).to_parquet(folder/"suma_opadow.parquet")
            frame,info=load_processed_chorzow(root)
            self.assertEqual(frame.index[-1].day,3)  # Last partial bin on Jan 4 is excluded.
            self.assertTrue(frame.loc[frame.index.day==2,"energia_kwh"].isna().all())
            self.assertEqual(frame.loc[frame.index.day==1,"energia_kwh"].dropna().iloc[0],10)
            self.assertTrue(info["excludedLastPartialHour"])

    def test_weather_30_days_hourly_contract_and_cache(self):
        calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"weather.json"
            frame=load_weather_forecast(50.292298,18.954600,now=NOW,cache_path=path,request_json=lambda url:(calls.append(url) or weather_payload()))
            again=load_weather_forecast(50.292298,18.954600,now=NOW+pd.Timedelta(minutes=1),cache_path=path,request_json=lambda url:self.fail("Cache should avoid network"))
        self.assertEqual(len(calls),1)
        self.assertIn("models=ecmwf_ec46&",calls[0])
        self.assertNotIn("ensemble_mean",calls[0])
        self.assertEqual(frame.attrs["ensembleMember"],0)
        self.assertEqual(frame.attrs["ensembleAggregation"],"none")
        self.assertTrue(again.attrs["cacheHit"])
        self.assertEqual(frame.attrs["expectedHours"],720)
        self.assertEqual(frame.attrs["availableWeatherHours"],720)

    def test_empty_weather_and_invalid_units_are_rejected(self):
        for response in ({},{"hourly":{"time":[],"precipitation":[]}}, {**weather_payload(),"hourly_units":{"precipitation":"inch"}}):
            with self.assertRaises(ValueError):
                load_weather_forecast(50,19,now=NOW,request_json=lambda url:response)

    def test_missing_weather_day_is_not_filled_with_zero(self):
        frame=load_weather_forecast(50,19,now=NOW,request_json=lambda url:weather_payload(True))
        self.assertEqual(frame["suma_opadu_mm"].isna().sum(),24)
        self.assertEqual(frame.attrs["availableWeatherHours"],696)

    def test_weather_uses_exactly_one_member_without_averaging(self):
        payload=weather_payload()
        count=len(payload["hourly"]["time"])
        payload["hourly"]["precipitation_member01"]=[8.0]*count
        payload["hourly_units"]["precipitation_member01"]="mm"
        first=load_weather_forecast(50,19,now=NOW,request_json=lambda url:payload)
        second=load_weather_forecast(50,19,now=NOW,request_json=lambda url:payload,ensemble_member=1)
        self.assertTrue(first["suma_opadu_mm"].eq(0).all())
        self.assertTrue(second["suma_opadu_mm"].eq(8).all())
        self.assertEqual(second.attrs["weatherVariable"],"precipitation_member01")
        self.assertEqual(second.attrs["ensembleMember"],1)

    def test_missing_requested_member_does_not_fall_back(self):
        with self.assertRaises(ValueError):
            load_weather_forecast(50,19,now=NOW,request_json=lambda url:weather_payload(),ensemble_member=1)
        for member in (-1,51,True,0.5):
            with self.assertRaises(ValueError):
                load_weather_forecast(50,19,now=NOW,request_json=lambda url:weather_payload(),ensemble_member=member)

    def test_old_mean_cache_is_not_reused(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"weather.json"
            load_weather_forecast(50,19,now=NOW,cache_path=path,request_json=lambda url:weather_payload())
            saved=json.loads(path.read_text())
            saved["key"]["models"]="ecmwf_ec46_ensemble_mean"
            path.write_text(json.dumps(saved))
            calls=[]
            fresh=load_weather_forecast(50,19,now=NOW,cache_path=path,request_json=lambda url:(calls.append(url) or weather_payload()))
            self.assertEqual(len(calls),1)
            self.assertFalse(fresh.attrs["cacheHit"])

    def fixture_forecast(self):
        detail,_=backend.load_station("chorzow",backend.fingerprint("chorzow"))
        weather=load_weather_forecast(50,19,now=NOW,request_json=lambda url:weather_payload())
        models=[("prognoza",ConstantModel(10),["opad_0h"],None), ("q97",ConstantModel(15),["opad_0h"],None),
                ("energia_prognoza",EnergyModel(),["przepompowano"],None), ("energia_q97",ConstantModel(40),["przepompowano"],None)]
        return backend.predict_chorzow_future(detail,weather,models)

    def test_future_energy_uses_predicted_pumping_and_actual_fields_stay_empty(self):
        public,projected,info=self.fixture_forecast()
        self.assertEqual(len(public),720)
        self.assertTrue(public["przepompowano_m3"].isna().all())
        self.assertTrue(public["energia_kwh"].isna().all())
        self.assertTrue(public["energia_prognoza"].eq(30).all())
        self.assertTrue(public["data_kind"].eq("prediction").all())
        self.assertTrue(projected["energia_kwh"].eq(30).all())
        self.assertEqual(info["status"],"ok")

    def test_future_valuation_uses_predictions_and_preserves_missing_hours(self):
        future=self.fixture_forecast()
        future[1].iloc[0,future[1].columns.get_loc("energia_kwh")]=np.nan
        with patch.object(backend,"chorzow_forecast",return_value=future):
            scenarios=backend.cost_scenarios("chorzow",30,{"lowPrice":1,"basePrice":2,"highPrice":3},"forecast","energy")
        result=scenarios[1]["valuation"]
        self.assertEqual(result["totalCostZl"],719*30*2)
        self.assertFalse(result["complete"])
        self.assertEqual(result["pricedHours"],719)
        self.assertEqual(result["dataKind"],"prediction")
        self.assertEqual(len(result["daily"]),30)

    def test_chorzow_costs_support_dates_and_historical_predictions(self):
        times = pd.date_range("2026-01-01", periods=48, freq="h", tz="Europe/Warsaw")
        detail = pd.DataFrame({"przepompowano_m3": 4., "energia_kwh": 2., "opad_mm": 0.,
                               "prognoza": 5., "energia_prognoza": 3., "okres": "suchy",
                               "bazowe_m3": 4., "bazowe_kwh": 2.}, index=times)
        prices = {"lowPrice": 1, "basePrice": 2, "highPrice": 3}
        with patch.object(backend, "load_station", return_value=(detail, [])):
            measured = backend.cost_scenarios("chorzow", 14, prices, "history", "energy", "2026-01-02", "2026-01-02")
            predicted = backend.cost_scenarios("chorzow", 14, prices, "history", "energy", "2026-01-02", "2026-01-02", "predicted")
            for start, end in (("2026-01-03", "2026-01-01"), ("2025-12-31", "2026-01-01"), ("2026-01-01", None)):
                with self.assertRaises(ValueError):
                    backend.cost_scenarios("chorzow", 14, prices, "history", "energy", start, end)
        self.assertEqual(measured[1]["valuation"]["totalCostZl"], 96)
        self.assertEqual(measured[1]["valuation"]["pumpingM3"], 96)
        self.assertEqual(predicted[1]["valuation"]["totalCostZl"], 144)
        self.assertEqual(predicted[1]["valuation"]["energyKwh"], 72)
        self.assertEqual(predicted[1]["valuation"]["pumpingM3"], 120)
        self.assertEqual(predicted[1]["valuation"]["dataKind"], "prediction")
        self.assertEqual(predicted[1]["valuation"]["expectedHours"], 24)
        pd.testing.assert_series_equal(detail["energia_kwh"], pd.Series(2., index=times, name="energia_kwh"))

    def test_chorzow_future_cost_dates_use_only_selected_predicted_hours(self):
        future = self.fixture_forecast()
        start = pd.Timestamp(future[2]["start"]).tz_convert("Europe/Warsaw").strftime("%Y-%m-%d")
        with patch.object(backend, "chorzow_forecast", return_value=future):
            values = backend.cost_scenarios("chorzow", 30, {"lowPrice": 1, "basePrice": 2, "highPrice": 3},
                                            "forecast", "energy", start, start, "measured")
        result = values[1]["valuation"]
        self.assertEqual(result["energyKwh"], 24*30)
        self.assertEqual(result["pumpingM3"], 24*10)
        self.assertEqual(result["totalCostZl"], 24*30*2)
        self.assertEqual(result["dataKind"], "prediction")
        self.assertEqual(result["expectedHours"], 24)

    def test_weather_failure_leaves_real_history_available_without_fake_forecast(self):
        with patch.object(backend,"chorzow_forecast",side_effect=ValueError("Empty Open-Meteo response")):
            data=backend.dashboard("chorzow")
        self.assertEqual(data["forecast"],[])
        self.assertEqual(data["weatherForecast"]["status"],"unavailable")
        self.assertTrue(any(r["energia_kwh"] is not None for r in data["history"]))
        self.assertEqual(data["stations"][0]["id"],"chorzow")
        self.assertEqual(len(data["stations"]),1)


if __name__=="__main__": unittest.main()
