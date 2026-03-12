import io
from typing import Dict, List, Optional


def synthesize_wav_bytes_from_voice(
    voice_bundle: List,
    text: str,
    language: str = "pt",
    temperature: Optional[float] = None,
    speed: Optional[float] = None,
) -> bytes:
    import numpy
    import soundfile as sf

    model, gpt_cond_latent, speaker_embedding, default_temperature, default_speed = voice_bundle

    out = model.inference(
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        temperature=default_temperature if temperature is None else float(temperature),
        speed=default_speed if speed is None else float(speed),
    )

    buffer = io.BytesIO()
    waveform = numpy.asarray(out["wav"], dtype=numpy.float32)
    sf.write(buffer, waveform, 24000, format="WAV")
    return buffer.getvalue()


def synthesize_wav_bytes(
    voice_model: str,
    text: str,
    registry: Dict[str, Dict],
    voice_cache: Dict[str, List],
    language: str = "pt",
    temperature: Optional[float] = None,
    speed: Optional[float] = None,
) -> bytes:
    from src.app.profiles import get_or_load_voice

    voice_bundle = get_or_load_voice(voice_model=voice_model, registry=registry, voice_cache=voice_cache)
    return synthesize_wav_bytes_from_voice(
        voice_bundle=voice_bundle,
        text=text,
        language=language,
        temperature=temperature,
        speed=speed,
    )


def synthesize_to_file(
    voice_model: str,
    text: str,
    registry: Dict[str, Dict],
    voice_cache: Dict[str, List],
    output_wav_path: str = "output.wav",
    language: str = "pt",
    temperature: Optional[float] = None,
    speed: Optional[float] = None,
) -> str:
    audio_bytes = synthesize_wav_bytes(
        voice_model=voice_model,
        text=text,
        registry=registry,
        voice_cache=voice_cache,
        language=language,
        temperature=temperature,
        speed=speed,
    )

    with open(output_wav_path, "wb") as file:
        file.write(audio_bytes)

    return output_wav_path
