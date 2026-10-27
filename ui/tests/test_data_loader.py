"""Loaders from src work independently of UI and saved models."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import data_loader
import pandas as pd


class ProcessedDataLoaderTests(unittest.TestCase):
    def write_weteranow(self, root):
        paths = data_loader.weteranow_processed_paths(root)
        paths["hourly"].parent.mkdir(parents=True)
        times = pd.date_range("2026-01-01", periods=6, freq="h", tz="Europe/Warsaw")
        frame = pd.DataFrame({"przepompowano_m3": [0., 2., 3., 4., 5., 6.],
                              "energia_kwh": [0., 1., 2., 3., 4., 5.], "opad_mm": 0.}, index=times)
        frame.drop(times[2]).to_parquet(paths["hourly"])
        pd.DataFrame({"DataCzas": ["2026-01-01 05:00:04"]}).to_parquet(paths["scada"])
        pd.DataFrame({"Data_czas": times.tz_localize(None), "Wartosc_kWh": "1,5"}).to_parquet(paths["energy"])
        return paths, times

    def test_weteranow_does_not_need_models_or_ui_and_preserves_gaps_and_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, times = self.write_weteranow(root)
            frame, _ = data_loader.load_processed_weteranow(root)
            self.assertEqual(len(frame), 5)  # Last incomplete bin is excluded.
            self.assertEqual(frame.loc[times[0], "energia_kwh"], 0)
            self.assertEqual(frame.loc[times[3], "energia_kwh"], 3)  # Already kWh, no diff().
            self.assertTrue(frame.loc[times[2]].isna().all())
            self.assertFalse((root / "models").exists())

    def test_weteranow_missing_source_and_invalid_time_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(FileNotFoundError, "Weteranów"):
                data_loader.load_processed_weteranow(root)
            paths, _ = self.write_weteranow(root)
            frame = pd.read_parquet(paths["hourly"])
            for index in (frame.index.tz_localize(None), pd.DatetimeIndex([frame.index[0]] * len(frame))):
                frame.index = index
                frame.to_parquet(paths["hourly"])
                with self.assertRaisesRegex(ValueError, "unikalny indeks czasu ze strefą"):
                    data_loader.load_processed_weteranow(root)

    def test_chorzow_reuses_original_counter_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "data/processed/Chorzów"
            folder.mkdir(parents=True)
            times = pd.date_range("2026-01-01", periods=5, freq="h")
            for filename, column in (("scieki.parquet", "Ilosc_przepompowanych_sciekow"),
                                     ("energia.parquet", "Zuzycie_energii_elektrycznej")):
                pd.DataFrame({"Data_czas": times, column: [10, 15, 15, 20, 22],
                              "Sposób_pomiaru": "Wartość z pomiaru"}).to_parquet(folder / filename)
            pd.DataFrame({"Data_czas": times, "suma_opadu_mm": 0.}).to_parquet(folder / "suma_opadow.parquet")
            with patch.object(data_loader, "convert_meter_data", wraps=data_loader.convert_meter_data) as convert:
                frame, _ = data_loader.load_processed_chorzow(root)
            self.assertEqual(convert.call_count, 2)
            self.assertEqual(frame["energia_kwh"].dropna().tolist(), [5., 0., 5.])

    def test_existing_csv_loader_contract_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meter.csv"
            path.write_text("Data;Godzina;Wartosc\n01.01.2026;12:00:00;5\n", encoding="utf8")
            frame = data_loader.load_standard_file(path)
            self.assertEqual(frame.loc[0, "Data_czas"], pd.Timestamp("2026-01-01 12:00:00"))
            self.assertEqual(frame.loc[0, "Wartosc"], 5)


if __name__ == "__main__":
    unittest.main()
