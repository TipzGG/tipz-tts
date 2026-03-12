import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from unittest.mock import MagicMock
import csv

from src.app.dataset import (
    _current_text,
    _iter_words,
    _merge_timing_ranges,
    _plan_pre_asr_chunks,
    _review_row,
    _should_flush_chunk,
    _text_has_suspicious_chars,
    _would_exceed_limits,
    clear_gpu_cache,
    review_dataset,
)


class DatasetTest(unittest.TestCase):
    def test_iter_words(self):
        word1 = MagicMock()
        word2 = MagicMock()
        s1 = MagicMock()
        s2 = MagicMock()
        s1.words = [word1]
        s2.words = [word2]
        self.assertEqual(_iter_words([s1, s2]), [word1, word2])

    def test_clear_gpu_cache(self):
        torch_module = MagicMock()
        torch_module.cuda.is_available.return_value = True
        clear_gpu_cache(torch_module)
        torch_module.cuda.empty_cache.assert_called_once()

    def test_current_text_trims_edges(self):
        self.assertEqual(_current_text([" ola", " mundo "]), "ola mundo")

    def test_should_flush_chunk_on_punctuation(self):
        self.assertTrue(_should_flush_chunk("ola mundo.", 0.0, 1.0, 200, 11.0))

    def test_should_flush_chunk_on_text_limit(self):
        self.assertTrue(_should_flush_chunk("a" * 200, 0.0, 1.0, 200, 11.0))

    def test_should_flush_chunk_on_duration_limit(self):
        self.assertTrue(_should_flush_chunk("ola mundo", 0.0, 11.5, 200, 11.0))

    def test_would_exceed_limits_on_next_word_text(self):
        self.assertTrue(_would_exceed_limits("a" * 179, "bb", 0.0, 1.0, 180, 8.0))

    def test_would_exceed_limits_on_next_word_duration(self):
        self.assertTrue(_would_exceed_limits("ola", " mundo", 0.0, 8.1, 180, 8.0))

    def test_merge_timing_ranges(self):
        self.assertEqual(_merge_timing_ranges([(0, 1000), (1100, 2000), (2600, 3000)], 150), [(0, 2000), (2600, 3000)])

    def test_plan_pre_asr_chunks_splits_long_range(self):
        chunks = _plan_pre_asr_chunks(
            [(1000, 14000)],
            audio_duration_ms=15000,
            max_chunk_seconds=5.0,
            min_chunk_seconds=1.0,
            keep_silence_ms=0,
            merge_gap_ms=100,
        )
        self.assertEqual(chunks, [(1000, 6000), (6000, 11000), (11000, 14000)])

    def test_plan_pre_asr_chunks_falls_back_to_full_audio(self):
        chunks = _plan_pre_asr_chunks(
            [],
            audio_duration_ms=9000,
            max_chunk_seconds=5.0,
            min_chunk_seconds=1.0,
            keep_silence_ms=200,
            merge_gap_ms=100,
        )
        self.assertEqual(chunks, [(0, 9000)])

    def test_text_has_suspicious_chars(self):
        self.assertTrue(_text_has_suspicious_chars("ola mundo ™"))
        self.assertFalse(_text_has_suspicious_chars("ola mundo!"))

    @patch("src.app.dataset._safe_audio_duration_seconds", return_value=0.5)
    def test_review_row_flags_short_audio_and_missing_punctuation(self, _mock_duration):
        reviewed = _review_row(
            {"audio_file": "wavs/a.wav", "text": "ola mundo", "speaker_name": "beerschool"},
            "/tmp/metadata_train.csv",
            min_audio_seconds=1.0,
            max_audio_seconds=12.0,
            min_text_chars=8,
            max_text_chars=180,
            min_chars_per_second=4.0,
            max_chars_per_second=24.0,
        )
        self.assertEqual(reviewed["review_status"], "review")
        self.assertIn("audio_too_short", reviewed["flags"])
        self.assertIn("no_terminal_punctuation", reviewed["flags"])

    @patch("src.app.dataset._safe_audio_duration_seconds", return_value=3.0)
    def test_review_row_marks_clean_sample_as_keep(self, _mock_duration):
        reviewed = _review_row(
            {"audio_file": "wavs/a.wav", "text": "Ola mundo.", "speaker_name": "beerschool"},
            "/tmp/metadata_train.csv",
            min_audio_seconds=1.0,
            max_audio_seconds=12.0,
            min_text_chars=8,
            max_text_chars=180,
            min_chars_per_second=2.0,
            max_chars_per_second=24.0,
        )
        self.assertEqual(reviewed["review_status"], "keep")
        self.assertEqual(reviewed["flags"], "")
        self.assertEqual(reviewed["score"], 100)

    @patch("src.app.dataset._safe_audio_duration_seconds", return_value=3.0)
    def test_review_dataset_uses_auto_status_policy(self, _mock_duration):
        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "metadata.csv"
            metadata.write_text(
                "audio_file|text|speaker_name\nwavs/a.wav|ola, tudo bem, aqui, agora, hoje, ainda... ™  sem fim|speaker\n",
                encoding="utf-8",
            )
            review_csv = Path(tmp) / "review.csv"
            filtered_csv = Path(tmp) / "filtered.csv"
            review_dataset(
                str(metadata),
                str(review_csv),
                str(filtered_csv),
                auto_status_policy="strict",
            )
            with review_csv.open() as file:
                rows = list(csv.DictReader(file, delimiter="|"))
            self.assertEqual(rows[0]["auto_status"], "drop")

    @patch("src.app.dataset._safe_audio_duration_seconds", return_value=3.0)
    def test_auto_curate_dataset_splits_writes_auto_csvs(self, _mock_duration):
        from src.app.dataset import auto_curate_dataset_splits

        with tempfile.TemporaryDirectory() as tmp:
            train_csv = Path(tmp) / "train.csv"
            eval_csv = Path(tmp) / "eval.csv"
            train_csv.write_text("audio_file|text|speaker_name\nwavs/a.wav|Ola mundo.|speaker\n", encoding="utf-8")
            eval_csv.write_text("audio_file|text|speaker_name\nwavs/b.wav|Ola, tudo bem, aqui, agora|speaker\n", encoding="utf-8")
            outputs = auto_curate_dataset_splits(str(train_csv), str(eval_csv), tmp)
            self.assertTrue(Path(outputs["train_scored_csv"]).exists())
            self.assertTrue(Path(outputs["eval_scored_csv"]).exists())
            self.assertTrue(Path(outputs["train_auto_csv"]).exists())
            self.assertTrue(Path(outputs["eval_auto_csv"]).exists())


if __name__ == "__main__":
    unittest.main()
