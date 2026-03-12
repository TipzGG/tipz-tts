import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.app.pipeline import download_sources, run_pipeline, slugify


class PipelineTest(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify(" Silvio Santos! "), "silvio_santos")

    def test_run_pipeline_skip_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "voice": {"name": "Silvio", "language": "pt"},
                "workspace_dir": tmp,
                "sources": [{"url": "https://youtube.com/fake"}],
                "train": {"enabled": False},
            }
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text("{}", encoding="utf-8")

            with patch("src.app.pipeline.load_config", return_value=config), patch(
                "src.app.pipeline.download_sources", return_value=Path(tmp) / "data.csv"
            ), patch("src.app.pipeline.build_dataset", return_value=("/tmp/train.csv", "/tmp/eval.csv")), patch(
                "src.app.pipeline.auto_curate_dataset_splits",
                return_value={
                    "train_scored_csv": "/tmp/train_scored.csv",
                    "eval_scored_csv": "/tmp/eval_scored.csv",
                    "train_auto_csv": "/tmp/train_auto.csv",
                    "eval_auto_csv": "/tmp/eval_auto.csv",
                    "policy": "strict",
                },
            ), patch(
                "src.app.pipeline.train_gpt"
            ) as train_mock:
                run_pipeline(cfg_path)

            train_mock.assert_not_called()

    def test_download_sources_reuses_existing_wav_without_redownload(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            raw_dir = workspace_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            cached_wav = raw_dir / "silvio_001.wav"
            cached_wav.write_bytes(b"wav")

            config = {
                "voice": {"name": "Silvio", "language": "pt"},
                "sources": [{"url": "https://youtube.com/fake"}],
            }

            with patch("src.app.pipeline.download_youtube_video_to_mp3") as download_mock, patch(
                "src.app.pipeline.convert_mp3_to_wav"
            ) as convert_mock, patch("src.app.pipeline.trim_wav_file") as trim_mock, patch(
                "src.app.pipeline.os.rename"
            ) as rename_mock:
                csv_path = download_sources(config, workspace_dir)

            download_mock.assert_not_called()
            convert_mock.assert_not_called()
            trim_mock.assert_not_called()
            rename_mock.assert_not_called()

            with open(csv_path, "r", encoding="utf-8", newline="") as file:
                rows = list(csv.reader(file))
            self.assertEqual(rows, [[str(cached_wav), "Silvio", "pt"]])


if __name__ == "__main__":
    unittest.main()
