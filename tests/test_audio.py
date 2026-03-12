import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from src.app.audio import download_youtube_video_to_mp3, process_inputs_folder, process_single_source, trim_wav_file


class AudioTest(unittest.TestCase):
    def test_trim_wav_file(self):
        fake_module = MagicMock()
        fake_audio = MagicMock()
        fake_audio.__getitem__.return_value = fake_audio
        fake_module.AudioSegment.from_wav.return_value = fake_audio

        with patch.dict(sys.modules, {"pydub": fake_module}), patch("src.app.audio.os.remove") as remove_mock:
            result = trim_wav_file(1, 2, "source.wav", "target.wav")

        self.assertEqual(result, "target.wav")
        fake_audio.export.assert_called_once_with("target.wav", format="wav")
        remove_mock.assert_called_once_with("source.wav")

    def test_process_single_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.app.audio.download_youtube_video_to_mp3", return_value="/tmp/a.mp3"), patch(
                "src.app.audio.convert_mp3_to_wav", return_value=(f"{tmp}/final.wav", f"{tmp}/enh.wav")
            ), patch("src.app.audio.os.rename") as rename_mock:
                result = process_single_source(
                    url="https://youtube.com/fake",
                    filename="voice_001",
                    output_folder=tmp,
                    trim_start=0,
                    trim_end=0,
                )

            self.assertEqual(result, f"{tmp}/final.wav")
            rename_mock.assert_called_once_with(f"{tmp}/enh.wav", f"{tmp}/final.wav")
            self.assertTrue(os.path.exists(os.path.join(tmp, "config.log")))

    def test_download_uses_cookies_env(self):
        captured_opts = {}

        class FakeYDL:
            def __init__(self, opts):
                captured_opts.update(opts)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download=True):
                return {"title": "x", "ext": "webm"}

            def prepare_filename(self, info_dict):
                return "/tmp/x.webm"

        fake_module = types.SimpleNamespace(YoutubeDL=FakeYDL)
        with patch.dict(sys.modules, {"yt_dlp": fake_module}), patch.dict(
            os.environ,
            {"YT_DLP_COOKIES_FROM_BROWSER": "chrome", "YT_DLP_COOKIES_FILE": "/tmp/cookies.txt"},
            clear=False,
        ):
            output = download_youtube_video_to_mp3("https://youtube.com/fake", "/tmp")

        self.assertEqual(output, "/tmp/x.mp3")
        self.assertEqual(captured_opts["cookiesfrombrowser"], ("chrome",))
        self.assertEqual(captured_opts["cookiefile"], "/tmp/cookies.txt")

    def test_process_inputs_folder_creates_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = os.path.join(tmp, "inputs")
            output_dir = os.path.join(tmp, "outputs")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(os.path.join(input_dir, "nested"), exist_ok=True)
            open(os.path.join(input_dir, "a.m4a"), "w", encoding="utf-8").close()
            open(os.path.join(input_dir, "nested", "b.mp4"), "w", encoding="utf-8").close()
            open(os.path.join(input_dir, "ignore.txt"), "w", encoding="utf-8").close()

            with patch("src.app.audio.process_local_source", side_effect=["/tmp/a.wav", "/tmp/b.wav"]) as process_mock:
                processed_files, csv_path = process_inputs_folder(
                    input_dir=input_dir,
                    output_folder=output_dir,
                    speaker_name="silvio",
                    language="pt",
                    skip_enhancement=True,
                )

            self.assertEqual(processed_files, ["/tmp/a.wav", "/tmp/b.wav"])
            self.assertEqual(process_mock.call_count, 2)
            self.assertTrue(process_mock.call_args_list[0].kwargs["skip_enhancement"])
            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, "r", encoding="utf-8") as file:
                lines = [line.strip() for line in file.readlines() if line.strip()]
            self.assertEqual(len(lines), 2)

    def test_process_inputs_folder_without_supported_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = os.path.join(tmp, "inputs")
            os.makedirs(input_dir, exist_ok=True)
            open(os.path.join(input_dir, "a.txt"), "w", encoding="utf-8").close()

            with self.assertRaises(ValueError):
                process_inputs_folder(
                    input_dir=input_dir,
                    output_folder=os.path.join(tmp, "outputs"),
                    speaker_name="silvio",
                )


if __name__ == "__main__":
    unittest.main()
