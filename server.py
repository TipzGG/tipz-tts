import argparse
import io
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, Tuple

from src.app.inference import synthesize_wav_bytes
from src.app.profiles import (
    load_voice_registry,
    preload_voices,
    public_voice_view,
    resolve_voice_config,
    split_registry_by_status,
)

LOGGER = logging.getLogger("tipz-tts")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

MAX_TEXT_LENGTH = 1000
ALLOWED_FORMATS = {"wav", "mp3"}
ALLOWED_SAMPLE_RATES = {16000, 22050, 24000}


class APIError(Exception):
    def __init__(self, code: str, message: str, status: int):
        self.code = code
        self.message = message
        self.status = status
        super().__init__(message)


class FixedWindowRateLimiter:
    def __init__(self, rpm: int):
        self.rpm = max(0, rpm)
        self._lock = threading.Lock()
        self._buckets: Dict[str, Tuple[float, int]] = {}

    def allow(self, key: str) -> bool:
        if self.rpm <= 0:
            return True
        now = time.time()
        with self._lock:
            window_start, count = self._buckets.get(key, (now, 0))
            if now-window_start >= 60:
                self._buckets[key] = (now, 1)
                return True
            if count >= self.rpm:
                return False
            self._buckets[key] = (window_start, count+1)
            return True


def create_app(voices_config: str, preload_all: bool = True):
    try:
        from flask import Flask, Response, jsonify, request
    except ImportError as exc:
        raise RuntimeError("Flask is required to run the server. Install dependencies with `make install`.") from exc

    app = Flask(__name__)

    full_registry = load_voice_registry(voices_config)
    registry, disabled_registry = split_registry_by_status(full_registry)
    voice_cache = preload_voices(registry) if preload_all else {}
    timeout_seconds = float(os.getenv("TTS_INFERENCE_TIMEOUT_SECONDS", "45"))
    max_workers = max(1, int(os.getenv("TTS_MAX_WORKERS", "2")))
    rate_limiter = FixedWindowRateLimiter(int(os.getenv("TTS_RATE_LIMIT_RPM", "0")))
    executor = ThreadPoolExecutor(max_workers=max_workers)
    metrics_lock = threading.Lock()
    metrics: Dict[str, int] = {
        "tts_requests_total": 0,
        "tts_errors_total": 0,
        "tts_inference_inflight": 0,
    }

    def log_event(event: str, **fields: Any) -> None:
        LOGGER.info(json.dumps({"event": event, **fields}, ensure_ascii=True))

    def make_error(code: str, message: str, status: int, request_id: str):
        with metrics_lock:
            metrics["tts_errors_total"] += 1
        return jsonify({"code": code, "message": message, "request_id": request_id}), status

    def parse_voice_payload(payload: Dict[str, Any], legacy: bool = False) -> Dict[str, Any]:
        text = str(payload.get("text", "")).strip()
        if not text:
            raise APIError("INVALID_ARGUMENT", "text is required", 400)
        if len(text) > MAX_TEXT_LENGTH:
            raise APIError("TEXT_TOO_LARGE", f"text exceeds max length ({MAX_TEXT_LENGTH})", 413)

        if legacy:
            voice_id = str(payload.get("voice_model") or payload.get("voice") or "").strip()
        else:
            voice_id = str(payload.get("voice_id", "")).strip()
        if not voice_id:
            raise APIError("INVALID_ARGUMENT", "voice_id is required", 400)

        try:
            voice_cfg = resolve_voice_config(voice_id, registry)
        except KeyError as exc:
            raise APIError("VOICE_NOT_FOUND", f"unknown voice_id '{voice_id}'", 404) from exc

        language = str(payload.get("language") or voice_cfg.get("language_default") or "pt").strip()
        audio_format = str(payload.get("format", "wav")).strip().lower()
        if audio_format not in ALLOWED_FORMATS:
            raise APIError("UNSUPPORTED_FORMAT", f"unsupported format '{audio_format}'", 415)

        sample_rate = int(payload.get("sample_rate", 24000))
        if sample_rate not in ALLOWED_SAMPLE_RATES:
            raise APIError("INVALID_ARGUMENT", "sample_rate must be one of 16000, 22050, 24000", 400)

        speaking_rate = float(payload.get("speaking_rate", voice_cfg["speed"]))
        if not 0.7 <= speaking_rate <= 1.3:
            raise APIError("INVALID_ARGUMENT", "speaking_rate must be between 0.7 and 1.3", 400)

        temperature = float(payload.get("temperature", voice_cfg["temperature"]))
        if not 0.1 <= temperature <= 1.2:
            raise APIError("INVALID_ARGUMENT", "temperature must be between 0.1 and 1.2", 400)

        return {
            "voice_id": voice_id,
            "text": text,
            "language": language,
            "format": audio_format,
            "sample_rate": sample_rate,
            "speaking_rate": speaking_rate,
            "temperature": temperature,
        }

    def convert_wav_bytes(wav_bytes: bytes, out_format: str, sample_rate: int) -> bytes:
        if out_format == "wav" and sample_rate == 24000:
            return wav_bytes
        try:
            import torch
            import torchaudio

            source_tensor, source_sr = torchaudio.load(io.BytesIO(wav_bytes))
            if source_sr != sample_rate:
                source_tensor = torchaudio.functional.resample(source_tensor, source_sr, sample_rate)
            output = io.BytesIO()
            file_format = "wav" if out_format == "wav" else "mp3"
            torchaudio.save(output, source_tensor, sample_rate, format=file_format)
            return output.getvalue()
        except Exception as exc:
            raise APIError("SYNTHESIS_FAILED", f"audio conversion failed: {exc}", 422) from exc

    def run_synthesis(validated: Dict[str, Any], request_id: str) -> bytes:
        started_at = time.time()
        with metrics_lock:
            metrics["tts_requests_total"] += 1
            metrics["tts_inference_inflight"] += 1
        try:
            future = executor.submit(
                synthesize_wav_bytes,
                voice_model=validated["voice_id"],
                text=validated["text"],
                registry=registry,
                voice_cache=voice_cache,
                language=validated["language"],
                temperature=validated["temperature"],
                speed=validated["speaking_rate"],
            )
            raw_wav = future.result(timeout=timeout_seconds)
            output = convert_wav_bytes(raw_wav, validated["format"], validated["sample_rate"])
            latency_ms = int((time.time() - started_at) * 1000)
            log_event(
                "tts.request.completed",
                request_id=request_id,
                voice_id=validated["voice_id"],
                format=validated["format"],
                text_length=len(validated["text"]),
                status=200,
                latency_ms=latency_ms,
            )
            return output
        except TimeoutError as exc:
            raise APIError("SYNTHESIS_FAILED", "synthesis timeout", 422) from exc
        except APIError:
            raise
        except Exception as exc:
            raise APIError("SYNTHESIS_FAILED", str(exc), 422) from exc
        finally:
            with metrics_lock:
                metrics["tts_inference_inflight"] = max(0, metrics["tts_inference_inflight"]-1)

    def handle_synthesis(legacy: bool = False):
        request_id = (
            str(request.headers.get("X-Request-Id", "")).strip()
            or str((request.get_json(silent=True) or {}).get("request_id", "")).strip()
            or str(uuid.uuid4())
        )
        client_key = request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown"
        if not rate_limiter.allow(client_key):
            return make_error("RATE_LIMITED", "too many requests", 429, request_id)

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return make_error("INVALID_ARGUMENT", "invalid json body", 400, request_id)

        try:
            validated = parse_voice_payload(payload, legacy=legacy)
            audio_bytes = run_synthesis(validated, request_id)
            mimetype = "audio/wav" if validated["format"] == "wav" else "audio/mpeg"
            response = Response(audio_bytes, mimetype=mimetype)
            response.headers["X-Voice-Id"] = validated["voice_id"]
            response.headers["X-Request-Id"] = request_id
            return response
        except APIError as exc:
            return make_error(exc.code, exc.message, exc.status, request_id)
        except Exception:
            return make_error("INTERNAL_ERROR", "internal server error", 500, request_id)

    @app.get("/health")
    @app.get("/healthz")
    def health():
        with metrics_lock:
            snapshot = dict(metrics)
        return jsonify(
            {
                "status": "ok",
                "loaded_voices": sorted(list(voice_cache.keys())),
                "available_voices": sorted(list(registry.keys())),
                "disabled_voices": sorted(list(disabled_registry.keys())),
                "metrics": snapshot,
            }
        )

    @app.get("/voices")
    @app.get("/v1/voices")
    def voices():
        voices_payload = [public_voice_view(voice_id, cfg) for voice_id, cfg in sorted(registry.items())]
        if request.path == "/voices":
            return jsonify({"voices": [voice["voice_id"] for voice in voices_payload]})
        return jsonify({"voices": voices_payload})

    @app.get("/v1/voices/<voice_id>")
    def voice_by_id(voice_id: str):
        try:
            cfg = resolve_voice_config(voice_id, registry)
        except KeyError:
            return jsonify({"code": "VOICE_NOT_FOUND", "message": "voice not found"}), 404
        return jsonify({"voice": public_voice_view(voice_id, cfg)})

    @app.post("/")
    @app.post("/tts")
    def tts():
        return handle_synthesis(legacy=True)

    @app.post("/v1/tts/synthesize")
    def synthesize_v1():
        return handle_synthesis(legacy=False)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tipz-tts HTTP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--voices-config", default="config/voices.json")
    parser.add_argument("--lazy-load", action="store_true", help="Load voices on first request instead of startup")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(voices_config=args.voices_config, preload_all=not args.lazy_load)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
