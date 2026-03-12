import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from src.app.audio import convert_mp3_to_wav, download_youtube_video_to_mp3, trim_wav_file
from src.app.dataset import build_dataset
from src.app.training import train_gpt


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def load_config(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    with open(path, "r", encoding="utf-8") as file:
        if suffix == ".json":
            config = json.load(file)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required for YAML configs. Install with: pip install pyyaml") from exc
            config = yaml.safe_load(file)
        else:
            raise ValueError("Unsupported config extension. Use .json (recommended) or .yaml/.yml")

    if not isinstance(config, dict):
        raise ValueError("Config root must be a YAML object")
    return config


def download_sources(config: dict, workspace_dir: Path) -> Path:
    voice_name = config["voice"]["name"]
    language = config["voice"].get("language", "pt")
    sources = config.get("sources", [])

    if not sources:
        raise ValueError("No sources found in YAML. Add at least one source.")

    raw_dir = workspace_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = workspace_dir / "data.csv"
    rows = []

    for index, source in enumerate(sources, start=1):
        url = source.get("url")
        if not url:
            raise ValueError(f"Source #{index} is missing 'url'")

        filename = source.get("filename") or f"{slugify(voice_name)}_{index:03d}"
        trim_start = int(source.get("trim_start", 0))
        trim_end = int(source.get("trim_end", 0))

        mp3_file = download_youtube_video_to_mp3(url, str(raw_dir))
        if not mp3_file:
            raise RuntimeError(f"Failed to download source #{index}: {url}")

        converted = convert_mp3_to_wav(mp3_file, str(raw_dir), filename)
        if not converted:
            raise RuntimeError(f"Failed to convert source #{index}: {url}")

        final_wav_path, enhanced_wav_path = converted

        if trim_end > 0:
            trimmed = trim_wav_file(trim_start, trim_end, enhanced_wav_path, final_wav_path)
            if not trimmed:
                raise RuntimeError(f"Failed to trim source #{index}: {url}")
        else:
            os.rename(enhanced_wav_path, final_wav_path)

        rows.append([final_wav_path, voice_name, language])

    with open(csv_path, "w", newline="") as csv_file:
        csv.writer(csv_file).writerows(rows)

    return csv_path


def run_pipeline(config_path: Path) -> None:
    config = load_config(config_path)

    voice = config.get("voice", {})
    voice_name = voice.get("name")
    language = voice.get("language", "pt")
    if not voice_name:
        raise ValueError("voice.name is required")

    workspace = config.get("workspace_dir") or f"outputs/{slugify(voice_name)}"
    workspace_dir = Path(workspace).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    source_csv = download_sources(config, workspace_dir)

    dataset_cfg = config.get("dataset", {})
    dataset_output_dir = Path(dataset_cfg.get("output_dir", str(workspace_dir / "dataset"))).resolve()

    train_csv, eval_csv = build_dataset(
        base_dataset=str(source_csv),
        output_dir=str(dataset_output_dir),
        val_split=float(dataset_cfg.get("val_split", 0.15)),
        buffer_seconds=float(dataset_cfg.get("buffer_seconds", 0.2)),
        whisper_model_size=dataset_cfg.get("whisper_model", "large-v2"),
        compute_type=dataset_cfg.get("compute_type", "float32"),
    )

    print(f"Dataset ready: {dataset_output_dir}")
    print(f"Train CSV: {train_csv}")
    print(f"Eval CSV: {eval_csv}")

    train_cfg = config.get("train", {})
    if not train_cfg.get("enabled", False):
        print("Train step skipped (train.enabled=false)")
        return

    max_audio_seconds = int(train_cfg.get("max_audio_seconds", 11))
    train_gpt(
        language=language,
        num_epochs=int(train_cfg.get("num_epochs", 10)),
        batch_size=int(train_cfg.get("batch_size", 4)),
        grad_acumm=int(train_cfg.get("grad_acumm", 1)),
        train_csv=train_csv,
        eval_csv=eval_csv,
        output_path=train_cfg.get("output_dir", str(workspace_dir / "training")),
        max_audio_length=int(max_audio_seconds * 22050),
    )

    print("Training finished.")
