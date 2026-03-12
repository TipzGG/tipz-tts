import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple


REQUIRED_VOICE_FIELDS = [
    "config_path",
    "tokenizer_path",
    "xtts_checkpoint",
    "speaker_reference",
]


def _normalize_voice_key(value: str) -> str:
    return value.strip().lower()


def load_voice_registry(registry_path: str = "config/voices.json") -> Dict[str, Dict]:
    path = Path(registry_path)
    if not path.exists():
        raise FileNotFoundError(f"Voice registry not found: {registry_path}. Create it from config/voices.example.json")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    voices = payload.get("voices") if isinstance(payload, dict) else None
    if not isinstance(voices, list) or not voices:
        raise ValueError("Voice registry must have a non-empty 'voices' array")

    normalized: Dict[str, Dict] = {}
    for item in voices:
        if not isinstance(item, dict):
            raise ValueError("Each voice item must be an object")

        voice_key = str(item.get("voice_model", "") or item.get("voice_id", "")).strip()
        config = item
        if not voice_key:
            raise ValueError("Each voice item must define 'voice_model' (voice_id)")

        if not isinstance(config, dict):
            raise ValueError(f"Voice config for '{voice_key}' must be an object")

        normalized_key = _normalize_voice_key(voice_key)
        status = str(config.get("status", "enabled")).strip().lower()
        enabled = status == "enabled"
        missing = [field for field in REQUIRED_VOICE_FIELDS if not config.get(field)]
        if missing and enabled:
            raise ValueError(f"Voice '{voice_key}' missing required fields: {', '.join(missing)}")
        if missing and not enabled:
            logging.getLogger(__name__).warning(
                "Skipping strict validation for disabled voice '%s': missing fields: %s",
                voice_key,
                ", ".join(missing),
            )
        normalized[normalized_key] = {
            "voice_id": voice_key,
            "display_name": str(config.get("display_name", voice_key)),
            "language_default": str(config.get("language_default", config.get("language", "pt"))),
            "config_path": str(config.get("config_path", "")),
            "tokenizer_path": str(config.get("tokenizer_path", "")),
            "xtts_checkpoint": str(config.get("xtts_checkpoint", "")),
            "speaker_reference": str(config.get("speaker_reference", "")),
            "temperature": float(config.get("temperature_default", config.get("temperature", 0.8))),
            "speed": float(config.get("speed_default", config.get("speed", 0.97))),
            "status": status,
            "updated_at": str(config.get("updated_at", "")),
        }

    return normalized


def split_registry_by_status(registry: Dict[str, Dict]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    enabled: Dict[str, Dict] = {}
    disabled: Dict[str, Dict] = {}
    for key, cfg in registry.items():
        if str(cfg.get("status", "enabled")).lower() == "enabled":
            enabled[key] = cfg
        else:
            disabled[key] = cfg
    return enabled, disabled


def public_voice_view(voice_id: str, cfg: Dict) -> Dict:
    return {
        "voice_id": voice_id,
        "display_name": cfg.get("display_name") or voice_id,
        "language_default": cfg.get("language_default", "pt"),
        "status": cfg.get("status", "enabled"),
        "updated_at": cfg.get("updated_at", ""),
    }


def resolve_voice_config(voice_model: str, registry: Dict[str, Dict]) -> Dict:
    key = _normalize_voice_key(voice_model)
    if key not in registry:
        raise KeyError(f"Unknown voice_model '{voice_model}'")
    return registry[key]


def load_voice_from_config(voice_config: Dict) -> List:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(voice_config["config_path"])

    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=voice_config["xtts_checkpoint"],
        vocab_path=voice_config["tokenizer_path"],
        use_deepspeed=False,
    )
    model.cpu()

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[voice_config["speaker_reference"]]
    )

    return [
        model,
        gpt_cond_latent,
        speaker_embedding,
        float(voice_config.get("temperature", 0.8)),
        float(voice_config.get("speed", 0.97)),
    ]


def get_or_load_voice(voice_model: str, registry: Dict[str, Dict], voice_cache: Dict[str, List]) -> List:
    key = _normalize_voice_key(voice_model)
    if key in voice_cache:
        return voice_cache[key]

    voice_config = resolve_voice_config(voice_model, registry)
    voice_cache[key] = load_voice_from_config(voice_config)
    return voice_cache[key]


def preload_voices(registry: Dict[str, Dict]) -> Dict[str, List]:
    cache: Dict[str, List] = {}
    for voice_key in registry.keys():
        cache[voice_key] = load_voice_from_config(registry[voice_key])
    return cache
