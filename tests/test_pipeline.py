import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.app.pipeline import run_pipeline, slugify


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
                "src.app.pipeline.train_gpt"
            ) as train_mock:
                run_pipeline(cfg_path)

            train_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
