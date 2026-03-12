import json
import tempfile
import unittest
from pathlib import Path

from src.app.profiles import load_voice_registry, resolve_voice_config


class ProfilesTest(unittest.TestCase):
    def test_load_voice_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voices.json"
            path.write_text(
                json.dumps(
                    {
                        "voices": [
                            {
                                "voice_model": "silvio",
                                "config_path": "/tmp/config.json",
                                "tokenizer_path": "/tmp/vocab.json",
                                "xtts_checkpoint": "/tmp/model.pth",
                                "speaker_reference": "/tmp/ref.wav",
                                "temperature": 0.85,
                                "speed": 0.99,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            registry = load_voice_registry(str(path))

        self.assertIn("silvio", registry)
        self.assertEqual(registry["silvio"]["temperature"], 0.85)
        self.assertEqual(registry["silvio"]["status"], "enabled")

    def test_load_voice_registry_allows_disabled_without_runtime_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voices.json"
            path.write_text(
                json.dumps(
                    {
                        "voices": [
                            {
                                "voice_model": "legacy",
                                "status": "disabled",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            registry = load_voice_registry(str(path))
        self.assertIn("legacy", registry)
        self.assertEqual(registry["legacy"]["status"], "disabled")

    def test_load_voice_registry_rejects_enabled_missing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voices.json"
            path.write_text(
                json.dumps(
                    {
                        "voices": [
                            {
                                "voice_model": "broken",
                                "status": "enabled",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_voice_registry(str(path))

    def test_resolve_voice_config(self):
        cfg = resolve_voice_config("SILVIO", {"silvio": {"speed": 0.97}})
        self.assertEqual(cfg["speed"], 0.97)


if __name__ == "__main__":
    unittest.main()
