import unittest
import wave
from io import BytesIO
from importlib.util import find_spec
from unittest.mock import patch

from server import create_app


@unittest.skipUnless(find_spec("flask") is not None, "flask is not installed")
class ServerTest(unittest.TestCase):
    @staticmethod
    def _valid_wav_bytes() -> bytes:
        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x00\x00" * 240)
        return buffer.getvalue()

    @staticmethod
    def _registry():
        return {
            "silvio": {
                "voice_id": "silvio",
                "display_name": "Silvio",
                "language_default": "pt",
                "status": "enabled",
                "speed": 0.97,
                "temperature": 0.8,
            },
            "legacy_disabled": {
                "voice_id": "legacy_disabled",
                "display_name": "Legacy Disabled",
                "language_default": "pt",
                "status": "disabled",
                "speed": 0.97,
                "temperature": 0.8,
            },
        }

    def test_health(self):
        with patch("server.load_voice_registry", return_value=self._registry()), patch(
            "server.split_registry_by_status",
            return_value=({"silvio": self._registry()["silvio"]}, {"legacy_disabled": self._registry()["legacy_disabled"]}),
        ), patch("server.preload_voices", return_value={"silvio": [1, 2, 3, 0.8, 0.97]}):
            app = create_app("/tmp/voices.json")
        client = app.test_client()
        res = client.get("/healthz")
        self.assertEqual(res.status_code, 200)

    def test_tts_requires_text_legacy(self):
        with patch("server.load_voice_registry", return_value=self._registry()), patch(
            "server.split_registry_by_status", return_value=({"silvio": self._registry()["silvio"]}, {})
        ), patch("server.preload_voices", return_value={"silvio": [1, 2, 3, 0.8, 0.97]}):
            app = create_app("/tmp/voices.json")
        client = app.test_client()
        res = client.post("/tts", json={"voice_model": "silvio"})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json["code"], "INVALID_ARGUMENT")

    def test_tts_success_legacy(self):
        with patch("server.load_voice_registry", return_value=self._registry()), patch(
            "server.split_registry_by_status", return_value=({"silvio": self._registry()["silvio"]}, {})
        ), patch("server.preload_voices", return_value={"silvio": [1, 2, 3, 0.8, 0.97]}), patch(
            "server.synthesize_wav_bytes", return_value=self._valid_wav_bytes()
        ):
            app = create_app("/tmp/voices.json")
        client = app.test_client()
        res = client.post("/tts", json={"text": "oi", "voice_model": "silvio"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content_type, "audio/wav")
        self.assertEqual(res.headers["X-Voice-Id"], "silvio")

    def test_v1_requires_voice_id(self):
        with patch("server.load_voice_registry", return_value=self._registry()), patch(
            "server.split_registry_by_status", return_value=({"silvio": self._registry()["silvio"]}, {})
        ), patch("server.preload_voices", return_value={"silvio": [1, 2, 3, 0.8, 0.97]}):
            app = create_app("/tmp/voices.json")
        client = app.test_client()
        res = client.post("/v1/tts/synthesize", json={"text": "oi"})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json["code"], "INVALID_ARGUMENT")

    def test_v1_voice_not_found(self):
        with patch("server.load_voice_registry", return_value=self._registry()), patch(
            "server.split_registry_by_status", return_value=({"silvio": self._registry()["silvio"]}, {})
        ), patch("server.preload_voices", return_value={"silvio": [1, 2, 3, 0.8, 0.97]}):
            app = create_app("/tmp/voices.json")
        client = app.test_client()
        res = client.post("/v1/tts/synthesize", json={"text": "oi", "voice_id": "not-found"})
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.json["code"], "VOICE_NOT_FOUND")

    def test_v1_lists_only_enabled_voices(self):
        with patch("server.load_voice_registry", return_value=self._registry()), patch(
            "server.split_registry_by_status",
            return_value=({"silvio": self._registry()["silvio"]}, {"legacy_disabled": self._registry()["legacy_disabled"]}),
        ), patch("server.preload_voices", return_value={"silvio": [1, 2, 3, 0.8, 0.97]}):
            app = create_app("/tmp/voices.json")
        client = app.test_client()
        res = client.get("/v1/voices")
        self.assertEqual(res.status_code, 200)
        voices = res.json["voices"]
        self.assertEqual(len(voices), 1)
        self.assertEqual(voices[0]["voice_id"], "silvio")


if __name__ == "__main__":
    unittest.main()
