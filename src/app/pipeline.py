import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from src.app.audio import convert_mp3_to_wav, download_youtube_video_to_mp3, trim_wav_file
from src.app.dataset import auto_curate_dataset_splits, build_dataset
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


def _source_cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


def _load_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)


def download_sources(config: dict, workspace_dir: Path) -> Path:
    voice_name = config["voice"]["name"]
    language = config["voice"].get("language", "pt")
    sources = config.get("sources", [])

    if not sources:
        raise ValueError("No sources found in YAML. Add at least one source.")

    raw_dir = workspace_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = workspace_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    sources_manifest_path = manifests_dir / "sources.json"
    sources_manifest = _load_json_file(sources_manifest_path, {"sources": []})
    manifest_index = {item.get("cache_key"): item for item in sources_manifest.get("sources", []) if item.get("cache_key")}

    csv_path = workspace_dir / "data.csv"
    rows = []

    for index, source in enumerate(sources, start=1):
        url = source.get("url")
        if not url:
            raise ValueError(f"Source #{index} is missing 'url'")

        filename = source.get("filename") or f"{slugify(voice_name)}_{index:03d}"
        trim_start = int(source.get("trim_start", 0))
        trim_end = int(source.get("trim_end", 0))
        cache_key = source.get("cache_key") or _source_cache_key(url)
        force_download = bool(source.get("force_download", False) or config.get("force_download", False))
        final_wav_path = str(raw_dir / f"{filename}.wav")

        cached_source = manifest_index.get(cache_key, {})
        cached_wav_path = str(cached_source.get("final_wav_path", "")).strip()
        if not force_download and cached_wav_path and os.path.exists(cached_wav_path):
            final_wav_path = cached_wav_path
        elif not force_download and os.path.exists(final_wav_path):
            pass
        else:
            mp3_file = download_youtube_video_to_mp3(url, str(raw_dir))
            if not mp3_file:
                raise RuntimeError(f"Failed to download source #{index}: {url}")

            converted = convert_mp3_to_wav(mp3_file, str(raw_dir), filename)
            if not converted:
                raise RuntimeError(f"Failed to convert source #{index}: {url}")

            generated_final_wav_path, enhanced_wav_path = converted

            if trim_end > 0:
                trimmed = trim_wav_file(trim_start, trim_end, enhanced_wav_path, generated_final_wav_path)
                if not trimmed:
                    raise RuntimeError(f"Failed to trim source #{index}: {url}")
                final_wav_path = trimmed
            else:
                os.rename(enhanced_wav_path, generated_final_wav_path)
                final_wav_path = generated_final_wav_path

        manifest_index[cache_key] = {
            "cache_key": cache_key,
            "url": url,
            "filename": filename,
            "trim_start": trim_start,
            "trim_end": trim_end,
            "final_wav_path": final_wav_path,
        }

        rows.append([final_wav_path, voice_name, language])

    with open(csv_path, "w", newline="") as csv_file:
        csv.writer(csv_file).writerows(rows)

    _write_json_file(
        sources_manifest_path,
        {"sources": [manifest_index[key] for key in sorted(manifest_index.keys())]},
    )

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
        max_segment_seconds=float(dataset_cfg.get("max_segment_seconds", 11.0)),
        max_text_chars=int(dataset_cfg.get("max_text_chars", 200)),
        pre_asr_max_chunk_seconds=float(dataset_cfg.get("pre_asr_max_chunk_seconds", 45.0)),
        pre_asr_min_chunk_seconds=float(dataset_cfg.get("pre_asr_min_chunk_seconds", 2.0)),
        pre_asr_min_silence_ms=int(dataset_cfg.get("pre_asr_min_silence_ms", 700)),
        pre_asr_keep_silence_ms=int(dataset_cfg.get("pre_asr_keep_silence_ms", 200)),
        pre_asr_merge_gap_ms=int(dataset_cfg.get("pre_asr_merge_gap_ms", 250)),
        pre_asr_silence_thresh_db=int(dataset_cfg.get("pre_asr_silence_thresh_db", -40)),
    )

    print(f"Dataset ready: {dataset_output_dir}")
    print(f"Train CSV: {train_csv}")
    print(f"Eval CSV: {eval_csv}")

    auto_cfg = dataset_cfg.get("auto_curate", {})
    auto_curate_enabled = bool(auto_cfg.get("enabled", True))
    auto_outputs = None
    if auto_curate_enabled:
        auto_outputs = auto_curate_dataset_splits(
            train_csv=train_csv,
            eval_csv=eval_csv,
            output_dir=str(dataset_output_dir),
            policy=str(auto_cfg.get("policy", "strict")),
            min_audio_seconds=float(auto_cfg.get("min_audio_seconds", 1.0)),
            max_audio_seconds=float(auto_cfg.get("max_audio_seconds", 12.0)),
            min_text_chars=int(auto_cfg.get("min_text_chars", 8)),
            max_text_chars=int(auto_cfg.get("max_text_chars", 180)),
            min_chars_per_second=float(auto_cfg.get("min_chars_per_second", 4.0)),
            max_chars_per_second=float(auto_cfg.get("max_chars_per_second", 24.0)),
        )
        print(f"Train scored CSV: {auto_outputs['train_scored_csv']}")
        print(f"Eval scored CSV: {auto_outputs['eval_scored_csv']}")
        print(f"Train auto CSV: {auto_outputs['train_auto_csv']}")
        print(f"Eval auto CSV: {auto_outputs['eval_auto_csv']}")

    train_cfg = config.get("train", {})
    if not train_cfg.get("enabled", False):
        print("Train step skipped (train.enabled=false)")
        return

    max_audio_seconds = int(train_cfg.get("max_audio_seconds", 11))
    selected_train_csv = train_csv
    selected_eval_csv = eval_csv
    if auto_outputs and bool(train_cfg.get("use_auto_dataset", True)):
        selected_train_csv = auto_outputs["train_auto_csv"]
        selected_eval_csv = auto_outputs["eval_auto_csv"]

    train_gpt(
        language=language,
        num_epochs=int(train_cfg.get("num_epochs", 10)),
        batch_size=int(train_cfg.get("batch_size", 4)),
        grad_acumm=int(train_cfg.get("grad_acumm", 1)),
        train_csv=selected_train_csv,
        eval_csv=selected_eval_csv,
        output_path=train_cfg.get("output_dir", str(workspace_dir / "training")),
        max_audio_length=int(max_audio_seconds * 22050),
        max_text_length=int(train_cfg.get("max_text_chars", 200)),
        mixed_precision=not bool(train_cfg.get("no_mixed_precision", False)),
        precision=str(train_cfg.get("precision", "fp16")),
    )

    print("Training finished.")
