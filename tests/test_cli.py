import unittest
from types import ModuleType
from unittest.mock import patch

import cli


class CLITest(unittest.TestCase):
    def test_train_accepts_grad_accum_alias(self):
        argv = [
            "cli.py",
            "train",
            "--train-csv",
            "train.csv",
            "--eval-csv",
            "eval.csv",
            "--grad-accum",
            "8",
        ]
        with patch("sys.argv", argv):
            args = cli.parse_args()

        self.assertEqual(args.command, "train")
        self.assertEqual(args.grad_accumm, 8)

    def test_train_accepts_resume_args(self):
        argv = [
            "cli.py",
            "train",
            "--train-csv",
            "train.csv",
            "--eval-csv",
            "eval.csv",
            "--restore-path",
            "/tmp/checkpoint_150.pth",
            "--resume-latest",
        ]
        with patch("sys.argv", argv):
            args = cli.parse_args()

        self.assertEqual(args.restore_path, "/tmp/checkpoint_150.pth")
        self.assertTrue(args.resume_latest)

    def test_infer_dispatch(self):
        argv = [
            "cli.py",
            "infer",
            "--voice-model",
            "silvio",
            "--text",
            "teste",
            "--output",
            "out.wav",
            "--voices-config",
            "templates/voices.json",
        ]
        with patch("sys.argv", argv), patch("src.app.profiles.load_voice_registry", return_value={"silvio": {}}), patch(
            "src.app.profiles.preload_voices", return_value={"silvio": [1, 2, 3, 0.8, 0.97]}
        ), patch("src.app.inference.synthesize_to_file", return_value="out.wav") as mock_fn:
            cli.main()

        mock_fn.assert_called_once()

    def test_import_inputs_dispatch(self):
        argv = [
            "cli.py",
            "import-inputs",
            "--input-dir",
            "inputs",
            "--output-folder",
            "outputs/local",
            "--speaker",
            "silvio",
            "--skip-enhancement",
        ]
        with patch("sys.argv", argv), patch("src.app.audio.process_inputs_folder", return_value=(["/tmp/a.wav"], "/tmp/data_from_inputs.csv")) as mock_fn:
            cli.main()

        mock_fn.assert_called_once()
        self.assertTrue(mock_fn.call_args.kwargs["skip_enhancement"])

    def test_isolate_speaker_dispatch(self):
        argv = [
            "cli.py",
            "isolate-speaker",
            "--input-dir",
            "inputs",
            "--output-dir",
            "outputs/isolated",
            "--speaker",
            "beerschool",
            "--allow-overlap",
        ]
        stub_module = ModuleType("src.app.speaker_isolation")
        with patch("sys.argv", argv), patch.object(
            stub_module,
            "isolate_speaker_from_inputs",
            return_value=(["/tmp/clip.wav"], "/tmp/data_from_isolated_speaker.csv"),
            create=True,
        ) as mock_fn, patch.dict("sys.modules", {"src.app.speaker_isolation": stub_module}):
            cli.main()

        mock_fn.assert_called_once()
        self.assertFalse(mock_fn.call_args.kwargs["drop_overlaps"])


if __name__ == "__main__":
    unittest.main()
