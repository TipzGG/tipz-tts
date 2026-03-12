import gc
import os
from typing import Tuple


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


def train_gpt(
    language: str,
    num_epochs: int,
    batch_size: int,
    grad_acumm: int,
    train_csv: str,
    eval_csv: str,
    output_path: str,
    max_audio_length: int = 255995,
) -> Tuple[str, str, str, str, str]:
    from trainer import Trainer, TrainerArgs

    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.xtts_config import XttsAudioConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
    from TTS.tts.utils.speakers import SpeakerManager
    from TTS.utils.manage import ModelManager

    if not _csv_has_data_rows(eval_csv):
        print("Warning: eval CSV has no rows. Reusing train CSV for evaluation.")
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

    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=os.path.dirname(os.path.abspath(train_csv)),
        meta_file_train=os.path.basename(train_csv),
        meta_file_val=os.path.basename(eval_csv),
        language=language,
    )

    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        debug_loading_failures=False,
        max_wav_length=max_audio_length,
        max_text_length=200,
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
        batch_group_size=48,
        eval_batch_size=batch_size,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=100,
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
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
        TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=False, grad_accum_steps=grad_acumm),
        config,
        output_path=out_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    sample_lengths = [len(item["text"].split(" ")) for item in train_samples]
    speaker_ref = train_samples[sample_lengths.index(max(sample_lengths))]["audio_file"]
    trainer_out_path = trainer.output_path

    del model, trainer, train_samples, eval_samples
    gc.collect()

    return xtts_config_file, xtts_checkpoint, tokenizer_file, trainer_out_path, speaker_ref
