import csv
import gc
import os
import re
from pathlib import Path
from typing import Optional, Tuple


def _csv_has_data_rows(csv_path: str) -> bool:
    with open(csv_path, "r", encoding="utf-8") as file:
        lines = [line for line in file.readlines() if line.strip()]
    return len(lines) > 1


def _download_if_missing(model_manager, files: list, output_dir: str) -> None:
    missing = [file_path for file_path in files if not os.path.isfile(file_path)]
    if missing:
        model_manager._download_model_files(
            [f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/{os.path.basename(path)}" for path in missing],
            output_dir,
            progress_bar=True,
        )


def _prepare_text_limited_csv(
    *,
    source_csv: str,
    output_dir: str,
    max_text_length: int,
    label: str,
) -> Tuple[str, int, int]:
    source_path = Path(source_csv)
    if not source_path.exists():
        raise FileNotFoundError(f"CSV not found: {source_csv}")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_csv = target_dir / f"{source_path.stem}_{label}_max{max_text_length}.csv"

    kept_rows = []
    total_rows = 0
    dropped_rows = 0

    with source_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="|")
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise RuntimeError(f"CSV header missing: {source_csv}")
        if "text" not in fieldnames:
            raise RuntimeError(f"CSV missing 'text' column: {source_csv}")

        for row in reader:
            total_rows += 1
            text = str(row.get("text", "")).strip()
            if len(text) > max_text_length:
                dropped_rows += 1
                continue
            kept_rows.append(row)

    if not kept_rows:
        raise RuntimeError(
            f"No rows left after text-length filter in {source_csv} (limit={max_text_length}). "
            "Increase --max-text-chars or regenerate dataset."
        )

    with target_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="|")
        writer.writeheader()
        writer.writerows(kept_rows)

    return str(target_csv), dropped_rows, total_rows


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint_(\d+)\.pth$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def _find_latest_checkpoint(training_output_path: str) -> Optional[str]:
    training_root = Path(training_output_path).resolve() / "run" / "training"
    if not training_root.exists():
        return None

    checkpoint_files = list(training_root.glob("GPT_XTTS_FT-*/checkpoint_*.pth"))
    checkpoint_files.extend(training_root.glob("checkpoint_*.pth"))
    if not checkpoint_files:
        return None

    latest = sorted(checkpoint_files, key=lambda path: (path.stat().st_mtime, _checkpoint_step(path)))[-1]
    return str(latest)


def _resolve_restore_checkpoint(
    *,
    output_path: str,
    restore_path: Optional[str],
    resume_latest: bool,
) -> Optional[str]:
    if restore_path:
        resolved = str(Path(restore_path).resolve())
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Restore checkpoint not found: {resolved}")
        return resolved

    if not resume_latest:
        return None

    latest = _find_latest_checkpoint(output_path)
    if not latest:
        raise RuntimeError(
            "resume_latest enabled but no checkpoint found under "
            f"{Path(output_path).resolve() / 'run' / 'training'}"
        )
    return latest


def train_gpt(
    language: str,
    num_epochs: int,
    batch_size: int,
    grad_acumm: int,
    train_csv: str,
    eval_csv: str,
    output_path: str,
    max_audio_length: int = 255995,
    max_text_length: int = 200,
    mixed_precision: bool = True,
    precision: str = "fp16",
    restore_path: Optional[str] = None,
    resume_latest: bool = False,
    batch_group_size: int = 48,
    num_loader_workers: int = 2,
    num_eval_loader_workers: int = 2,
    save_step: int = 250,
    plot_step: int = 10**9,
    log_model_step: int = 10**9,
) -> Tuple[str, str, str, str, str]:
    import torch
    from trainer import Trainer, TrainerArgs
    from trainer.logging.dummy_logger import DummyLogger

    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.xtts_config import XttsAudioConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
    from TTS.tts.utils.speakers import SpeakerManager
    from TTS.utils.manage import ModelManager

    if not _csv_has_data_rows(eval_csv):
        print("Warning: eval CSV has no rows. Reusing train CSV for evaluation.")
        eval_csv = train_csv

    resolved_restore_path = _resolve_restore_checkpoint(
        output_path=output_path,
        restore_path=restore_path,
        resume_latest=resume_latest,
    )
    if resolved_restore_path:
        print(f"[train] resuming from checkpoint: {resolved_restore_path}")

    prepared_train_csv_dir = str(Path(train_csv).resolve().parent)
    prepared_eval_csv_dir = str(Path(eval_csv).resolve().parent)
    train_csv, dropped_train, total_train = _prepare_text_limited_csv(
        source_csv=train_csv,
        output_dir=prepared_train_csv_dir,
        max_text_length=max_text_length,
        label="train",
    )
    eval_csv, dropped_eval, total_eval = _prepare_text_limited_csv(
        source_csv=eval_csv,
        output_dir=prepared_eval_csv_dir,
        max_text_length=max_text_length,
        label="eval",
    )
    print(
        f"[train] text-length filter: train dropped {dropped_train}/{total_train}, "
        f"eval dropped {dropped_eval}/{total_eval}, limit={max_text_length}"
    )

    if not _csv_has_data_rows(eval_csv):
        print("Warning: filtered eval CSV has no rows. Reusing filtered train CSV for evaluation.")
        eval_csv = train_csv

    out_path = os.path.join(output_path, "run", "training")
    os.makedirs(out_path, exist_ok=True)

    checkpoints_out_path = os.path.join(out_path, "XTTS_v2.0_original_model_files")
    os.makedirs(checkpoints_out_path, exist_ok=True)

    dvae_checkpoint = os.path.join(checkpoints_out_path, "dvae.pth")
    mel_norm_file = os.path.join(checkpoints_out_path, "mel_stats.pth")
    tokenizer_file = os.path.join(checkpoints_out_path, "vocab.json")
    xtts_checkpoint = os.path.join(checkpoints_out_path, "model.pth")
    xtts_config_file = os.path.join(checkpoints_out_path, "config.json")

    _download_if_missing(ModelManager, [dvae_checkpoint, mel_norm_file], checkpoints_out_path)
    _download_if_missing(ModelManager, [tokenizer_file, xtts_checkpoint, xtts_config_file], checkpoints_out_path)

    max_conditioning_length = min(int(4 * 22050), max_audio_length)
    min_conditioning_length = min(int(2 * 22050), max_conditioning_length)

    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=os.path.dirname(os.path.abspath(train_csv)),
        meta_file_train=os.path.basename(train_csv),
        meta_file_val=os.path.basename(eval_csv),
        language=language,
    )

    model_args = GPTArgs(
        max_conditioning_length=max_conditioning_length,
        min_conditioning_length=min_conditioning_length,
        debug_loading_failures=False,
        max_wav_length=max_audio_length,
        max_text_length=max_text_length,
        mel_norm_file=mel_norm_file,
        dvae_checkpoint=dvae_checkpoint,
        xtts_checkpoint=xtts_checkpoint,
        tokenizer_file=tokenizer_file,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    config = GPTTrainerConfig(
        mixed_precision=bool(mixed_precision and torch.cuda.is_available()),
        precision=precision if torch.cuda.is_available() else "fp32",
        use_grad_scaler=bool(mixed_precision and torch.cuda.is_available()),
        epochs=num_epochs,
        output_path=out_path,
        model_args=model_args,
        run_name="GPT_XTTS_FT",
        project_name="XTTS_trainer",
        run_description="GPT XTTS training",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=16000),
        batch_size=batch_size,
        batch_group_size=batch_group_size,
        eval_batch_size=batch_size,
        num_loader_workers=num_loader_workers,
        num_eval_loader_workers=num_eval_loader_workers,
        eval_split_max_size=256,
        print_step=50,
        plot_step=plot_step,
        log_model_step=log_model_step,
        save_step=save_step,
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_eval=False,
        run_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2, "foreach": False},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
        max_text_len=max_text_length,
        max_audio_len=max_audio_length,
    )

    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
    config.speaker_manager = speaker_manager
    config.num_speakers = speaker_manager.num_speakers

    model = GPTTrainer.init_from_config(config)
    trainer = Trainer(
        TrainerArgs(
            restore_path=resolved_restore_path,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=grad_acumm,
        ),
        config,
        output_path=out_path,
        dashboard_logger=DummyLogger(),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.update_training_dashboard_logger = lambda *args, **kwargs: None
    trainer.fit()

    sample_lengths = [len(item["text"].split(" ")) for item in train_samples]
    speaker_ref = train_samples[sample_lengths.index(max(sample_lengths))]["audio_file"]
    trainer_out_path = trainer.output_path

    del model, trainer, train_samples, eval_samples
    gc.collect()

    return xtts_config_file, xtts_checkpoint, tokenizer_file, trainer_out_path, speaker_ref
