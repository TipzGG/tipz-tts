import tempfile
import unittest
import csv
from pathlib import Path

from src.app.training import _prepare_text_limited_csv, _resolve_restore_checkpoint


class TrainingTest(unittest.TestCase):
    def test_prepare_text_limited_csv_drops_long_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_csv = Path(tmp) / "train.csv"
            source_csv.write_text(
                "audio_file|text|speaker_name\n"
                "wavs/a.wav|texto curto.|spk\n"
                "wavs/b.wav|"
                + ("x" * 220)
                + "|spk\n",
                encoding="utf-8",
            )

            filtered_csv, dropped, total = _prepare_text_limited_csv(
                source_csv=str(source_csv),
                output_dir=tmp,
                max_text_length=180,
                label="train",
            )

            self.assertEqual(dropped, 1)
            self.assertEqual(total, 2)
            rows = Path(filtered_csv).read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 2)
            with Path(filtered_csv).open("r", encoding="utf-8") as file:
                parsed = list(csv.DictReader(file, delimiter="|"))
            self.assertEqual(parsed[0]["audio_file"], "wavs/a.wav")

    def test_prepare_text_limited_csv_raises_when_all_rows_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_csv = Path(tmp) / "train.csv"
            source_csv.write_text(
                "audio_file|text|speaker_name\n"
                "wavs/a.wav|"
                + ("x" * 250)
                + "|spk\n",
                encoding="utf-8",
            )

            with self.assertRaises(RuntimeError):
                _prepare_text_limited_csv(
                    source_csv=str(source_csv),
                    output_dir=tmp,
                    max_text_length=180,
                    label="train",
                )

    def test_resolve_restore_checkpoint_uses_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint_200.pth"
            checkpoint.write_bytes(b"")
            resolved = _resolve_restore_checkpoint(
                output_path=tmp,
                restore_path=str(checkpoint),
                resume_latest=False,
            )
            self.assertEqual(resolved, str(checkpoint.resolve()))

    def test_resolve_restore_checkpoint_finds_latest_when_resume_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run" / "training" / "GPT_XTTS_FT-test"
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_a = run_dir / "checkpoint_50.pth"
            ckpt_b = run_dir / "checkpoint_150.pth"
            ckpt_a.write_bytes(b"a")
            ckpt_b.write_bytes(b"b")
            resolved = _resolve_restore_checkpoint(
                output_path=tmp,
                restore_path=None,
                resume_latest=True,
            )
            self.assertEqual(resolved, str(ckpt_b.resolve()))


if __name__ == "__main__":
    unittest.main()
