import csv
import os
import re
import shutil
import sys
import types
from pathlib import Path
from typing import Optional, Tuple


class AudioDownloadError(RuntimeError):
    pass


def _ensure_torchaudio_backend_compat() -> None:
    import torchaudio

    if not hasattr(torchaudio, "info"):
        class _AudioMetaDataCompat:
            def __init__(self, sample_rate: int, num_frames: int, num_channels: int):
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = 16
                self.encoding = "UNKNOWN"

        def _info_compat(file: str, **kwargs):
            audio, sample_rate = torchaudio.load(file, **kwargs)
            num_channels = int(audio.shape[0]) if audio.ndim > 1 else 1
            num_frames = int(audio.shape[-1])
            return _AudioMetaDataCompat(sample_rate, num_frames, num_channels)

        torchaudio.info = _info_compat  # type: ignore[attr-defined]

    try:
        import torchaudio.backend.common  # type: ignore # noqa: F401
        return
    except Exception:
        pass

    # DeepFilterNet expects torchaudio.backend.common.AudioMetaData (removed in newer torchaudio releases).
    # This shim preserves import compatibility; ta.info() return object still carries required fields.
    backend_module = sys.modules.get("torchaudio.backend")
    if backend_module is None:
        backend_module = types.ModuleType("torchaudio.backend")
        sys.modules["torchaudio.backend"] = backend_module

    common_module = types.ModuleType("torchaudio.backend.common")
    common_module.AudioMetaData = object
    backend_module.common = common_module  # type: ignore[attr-defined]
    sys.modules["torchaudio.backend.common"] = common_module


def trim_wav_file(start_seconds: int, end_seconds: int, source_wav: str, target_wav: str) -> str:
    from pydub import AudioSegment

    audio = AudioSegment.from_wav(source_wav)
    trimmed_audio = audio[start_seconds * 1000 : end_seconds * 1000]
    trimmed_audio.export(target_wav, format="wav")
    os.remove(source_wav)
    return target_wav


def enhance_audio_file(input_path: str, output_path: str) -> str:
    _ensure_torchaudio_backend_compat()

    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
    except Exception as exc:
        print(f"Warning: enhancement unavailable ({exc}). Keeping original audio.")
        shutil.copyfile(input_path, output_path)
        return output_path

    try:
        model, df_state, _ = init_df()
        audio, _ = load_audio(input_path, sr=df_state.sr())
        enhanced = enhance(model, df_state, audio)
        save_audio(output_path, enhanced, df_state.sr())
    except Exception as exc:
        print(f"Warning: enhancement failed ({exc}). Keeping original audio.")
        shutil.copyfile(input_path, output_path)
    return output_path


def download_youtube_video_to_mp3(url: str, output_path: str = ".") -> Optional[str]:
    import yt_dlp

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_path, "%(title)s.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
    }
    cookies_from_browser = os.getenv("YT_DLP_COOKIES_FROM_BROWSER", "").strip()
    cookies_file = os.getenv("YT_DLP_COOKIES_FILE", "").strip()
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info_dict)
        return filename.replace(".webm", ".mp3").replace(".m4a", ".mp3")
    except Exception as exc:
        print(f"Failed to download {url}: {exc}")
        return None


def convert_mp3_to_wav(mp3_file: str, output_path: str = ".", filename: str = "output") -> Optional[Tuple[str, str]]:
    from pydub import AudioSegment

    try:
        original_wav = os.path.join(output_path, f"{filename}.wav")
        enhanced_wav = os.path.join(output_path, f"{filename}_enh.wav")
        preview_wav = os.path.join(output_path, f"{filename}_prev.wav")

        audio = AudioSegment.from_mp3(mp3_file)
        audio.export(preview_wav, format="wav")

        os.remove(mp3_file)
        enhance_audio_file(preview_wav, enhanced_wav)
        os.remove(preview_wav)

        return original_wav, enhanced_wav
    except Exception as exc:
        print(f"An error occurred during WAV conversion: {exc}")
        return None


def process_single_source(url: str, filename: str, output_folder: str, trim_start: int = 0, trim_end: int = 0) -> str:
    mp3_file = download_youtube_video_to_mp3(url, output_folder)
    if not mp3_file:
        raise AudioDownloadError("Failed to download source audio")

    converted = convert_mp3_to_wav(mp3_file, output_folder, filename)
    if not converted:
        raise AudioDownloadError("Failed to convert source audio")

    final_wav, enhanced_wav = converted

    if trim_end > 0:
        trim_wav_file(trim_start, trim_end, enhanced_wav, final_wav)
    else:
        os.rename(enhanced_wav, final_wav)

    with open(os.path.join(output_folder, "config.log"), "a", encoding="utf-8") as file:
        file.write(f"url: {url} | {final_wav} | {trim_start} - {trim_end}\n")

    return final_wav


def _slugify_file_stem(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "audio"


def process_local_source(
    input_file: str,
    output_folder: str,
    filename: str,
    trim_start: int = 0,
    trim_end: int = 0,
    skip_enhancement: bool = False,
) -> str:
    from pydub import AudioSegment

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_wav = output_dir / f"{filename}.wav"
    enhanced_wav = output_dir / f"{filename}_enh.wav"
    preview_wav = output_dir / f"{filename}_prev.wav"

    audio = AudioSegment.from_file(str(input_path))
    audio.export(str(preview_wav), format="wav")
    if skip_enhancement:
        os.rename(preview_wav, enhanced_wav)
    else:
        enhance_audio_file(str(preview_wav), str(enhanced_wav))
        os.remove(preview_wav)

    if trim_end > 0:
        trim_wav_file(trim_start, trim_end, str(enhanced_wav), str(final_wav))
    else:
        os.rename(enhanced_wav, final_wav)

    return str(final_wav)


def process_inputs_folder(
    input_dir: str,
    output_folder: str,
    speaker_name: str,
    language: str = "pt",
    trim_start: int = 0,
    trim_end: int = 0,
    skip_enhancement: bool = False,
) -> Tuple[list, str]:
    supported_extensions = {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".flac", ".ogg", ".aac"}
    source_dir = Path(input_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(
        file_path for file_path in source_dir.rglob("*") if file_path.is_file() and file_path.suffix.lower() in supported_extensions
    )
    if not files:
        raise ValueError(f"No supported audio/video files found in: {input_dir}")

    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    processed_files = []
    for index, file_path in enumerate(files, start=1):
        filename = f"{_slugify_file_stem(file_path.stem)}_{index:03d}"
        final_wav = process_local_source(
            input_file=str(file_path),
            output_folder=str(output_dir),
            filename=filename,
            trim_start=trim_start,
            trim_end=trim_end,
            skip_enhancement=skip_enhancement,
        )
        rows.append([final_wav, speaker_name, language])
        processed_files.append(final_wav)

    csv_path = output_dir / "data_from_inputs.csv"
    with open(csv_path, "w", newline="") as csv_file:
        csv.writer(csv_file).writerows(rows)

    return processed_files, str(csv_path)
