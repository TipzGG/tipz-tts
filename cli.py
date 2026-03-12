import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tipz-tts CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_download = subparsers.add_parser("download", help="Download and preprocess one YouTube source")
    p_download.add_argument("--url", required=True)
    p_download.add_argument("--filename", required=True)
    p_download.add_argument("--output-folder", default=".")
    p_download.add_argument("--trim-start", type=int, default=0)
    p_download.add_argument("--trim-end", type=int, default=0)

    p_enhance = subparsers.add_parser("enhance", help="Enhance a local WAV")
    p_enhance.add_argument("--input", required=True)
    p_enhance.add_argument("--output", default="enhanced.wav")

    p_dataset = subparsers.add_parser("dataset", help="Build dataset from CSV")
    p_dataset.add_argument("--input-csv", default="templates/data_example.csv")
    p_dataset.add_argument("--output-dir", default="output")
    p_dataset.add_argument("--val-split", type=float, default=0.15)
    p_dataset.add_argument("--buffer-seconds", type=float, default=0.2)
    p_dataset.add_argument("--whisper-model", default="large-v2")
    p_dataset.add_argument("--compute-type", default="float32")
    p_dataset.add_argument("--max-segment-seconds", type=float, default=11.0)
    p_dataset.add_argument("--max-text-chars", type=int, default=200)
    p_dataset.add_argument("--pre-asr-max-chunk-seconds", type=float, default=45.0)
    p_dataset.add_argument("--pre-asr-min-chunk-seconds", type=float, default=2.0)
    p_dataset.add_argument("--pre-asr-min-silence-ms", type=int, default=700)
    p_dataset.add_argument("--pre-asr-keep-silence-ms", type=int, default=200)
    p_dataset.add_argument("--pre-asr-merge-gap-ms", type=int, default=250)
    p_dataset.add_argument("--pre-asr-silence-thresh-db", type=int, default=-40)

    p_review_dataset = subparsers.add_parser("review-dataset", help="Flag suspicious dataset rows for manual curation")
    p_review_dataset.add_argument("--metadata-csv", required=True)
    p_review_dataset.add_argument("--output-csv")
    p_review_dataset.add_argument("--filtered-output-csv")
    p_review_dataset.add_argument("--min-audio-seconds", type=float, default=1.0)
    p_review_dataset.add_argument("--max-audio-seconds", type=float, default=12.0)
    p_review_dataset.add_argument("--min-text-chars", type=int, default=8)
    p_review_dataset.add_argument("--max-text-chars", type=int, default=180)
    p_review_dataset.add_argument("--min-chars-per-second", type=float, default=4.0)
    p_review_dataset.add_argument("--max-chars-per-second", type=float, default=24.0)

    p_train = subparsers.add_parser("train", help="Train XTTS")
    p_train.add_argument("--train-csv", required=True)
    p_train.add_argument("--eval-csv", required=True)
    p_train.add_argument("--language", default="pt")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--grad-accumm", type=int, default=1)
    p_train.add_argument("--max-audio-seconds", type=int, default=11)
    p_train.add_argument("--max-text-chars", type=int, default=200)
    p_train.add_argument("--output-dir", default="out")
    p_train.add_argument("--precision", default="fp16")
    p_train.add_argument("--no-mixed-precision", action="store_true")

    p_infer = subparsers.add_parser("infer", help="Run local inference")
    p_infer.add_argument("--voice-model", required=True)
    p_infer.add_argument("--text", required=True)
    p_infer.add_argument("--language", default="pt")
    p_infer.add_argument("--output", default="output.wav")
    p_infer.add_argument("--temperature", type=float)
    p_infer.add_argument("--speed", type=float)
    p_infer.add_argument("--voices-config", default="config/voices.json")

    p_pipeline = subparsers.add_parser("pipeline", help="Run config pipeline")
    p_pipeline.add_argument("--config", required=True)

    p_import_inputs = subparsers.add_parser("import-inputs", help="Process local files inside inputs folder")
    p_import_inputs.add_argument("--input-dir", default="inputs")
    p_import_inputs.add_argument("--output-folder", default="outputs/local")
    p_import_inputs.add_argument("--speaker", required=True)
    p_import_inputs.add_argument("--language", default="pt")
    p_import_inputs.add_argument("--trim-start", type=int, default=0)
    p_import_inputs.add_argument("--trim-end", type=int, default=0)
    p_import_inputs.add_argument("--skip-enhancement", action="store_true")

    p_isolate = subparsers.add_parser("isolate-speaker", help="Diarize and export only one speaker from local files")
    p_isolate.add_argument("--input-dir", default="inputs")
    p_isolate.add_argument("--output-dir", default="outputs/isolated")
    p_isolate.add_argument("--speaker", required=True)
    p_isolate.add_argument("--language", default="pt")
    p_isolate.add_argument("--target-speaker")
    p_isolate.add_argument("--reference-audio")
    p_isolate.add_argument("--hf-token")
    p_isolate.add_argument("--min-segment-seconds", type=float, default=1.5)
    p_isolate.add_argument("--max-segment-seconds", type=float, default=15.0)
    p_isolate.add_argument("--allow-overlap", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "download":
        from src.app.audio import process_single_source

        result = process_single_source(
            url=args.url,
            filename=args.filename,
            output_folder=args.output_folder,
            trim_start=args.trim_start,
            trim_end=args.trim_end,
        )
        print(f"Saved cleaned audio: {result}")
        return

    if args.command == "enhance":
        from src.app.audio import enhance_audio_file

        result = enhance_audio_file(args.input, args.output)
        print(f"Saved enhanced audio: {result}")
        return

    if args.command == "dataset":
        from src.app.dataset import build_dataset

        train_csv, eval_csv = build_dataset(
            base_dataset=args.input_csv,
            output_dir=args.output_dir,
            val_split=args.val_split,
            buffer_seconds=args.buffer_seconds,
            whisper_model_size=args.whisper_model,
            compute_type=args.compute_type,
            max_segment_seconds=args.max_segment_seconds,
            max_text_chars=args.max_text_chars,
            pre_asr_max_chunk_seconds=args.pre_asr_max_chunk_seconds,
            pre_asr_min_chunk_seconds=args.pre_asr_min_chunk_seconds,
            pre_asr_min_silence_ms=args.pre_asr_min_silence_ms,
            pre_asr_keep_silence_ms=args.pre_asr_keep_silence_ms,
            pre_asr_merge_gap_ms=args.pre_asr_merge_gap_ms,
            pre_asr_silence_thresh_db=args.pre_asr_silence_thresh_db,
        )
        print(f"Train metadata: {train_csv}")
        print(f"Eval metadata: {eval_csv}")
        return

    if args.command == "review-dataset":
        from src.app.dataset import review_dataset

        review_csv, filtered_csv = review_dataset(
            metadata_csv=args.metadata_csv,
            output_csv=args.output_csv,
            filtered_output_csv=args.filtered_output_csv,
            min_audio_seconds=args.min_audio_seconds,
            max_audio_seconds=args.max_audio_seconds,
            min_text_chars=args.min_text_chars,
            max_text_chars=args.max_text_chars,
            min_chars_per_second=args.min_chars_per_second,
            max_chars_per_second=args.max_chars_per_second,
        )
        print(f"Review metadata: {review_csv}")
        if filtered_csv:
            print(f"Filtered metadata: {filtered_csv}")
        return

    if args.command == "train":
        from src.app.training import train_gpt

        train_gpt(
            language=args.language,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            grad_acumm=args.grad_accumm,
            train_csv=args.train_csv,
            eval_csv=args.eval_csv,
            output_path=args.output_dir,
            max_audio_length=int(args.max_audio_seconds * 22050),
            max_text_length=args.max_text_chars,
            mixed_precision=not args.no_mixed_precision,
            precision=args.precision,
        )
        return

    if args.command == "infer":
        from src.app.inference import synthesize_to_file
        from src.app.profiles import load_voice_registry, preload_voices

        registry = load_voice_registry(args.voices_config)
        cache = preload_voices(registry)

        output = synthesize_to_file(
            voice_model=args.voice_model,
            text=args.text,
            language=args.language,
            output_wav_path=args.output,
            registry=registry,
            voice_cache=cache,
            temperature=args.temperature,
            speed=args.speed,
        )
        print(f"Saved: {output}")
        return

    if args.command == "pipeline":
        from pathlib import Path
        from src.app.pipeline import run_pipeline

        run_pipeline(Path(args.config).resolve())
        return

    if args.command == "import-inputs":
        from src.app.audio import process_inputs_folder

        processed_files, csv_path = process_inputs_folder(
            input_dir=args.input_dir,
            output_folder=args.output_folder,
            speaker_name=args.speaker,
            language=args.language,
            trim_start=args.trim_start,
            trim_end=args.trim_end,
            skip_enhancement=args.skip_enhancement,
        )
        print(f"Processed files: {len(processed_files)}")
        print(f"CSV ready: {csv_path}")
        return

    if args.command == "isolate-speaker":
        from src.app.speaker_isolation import isolate_speaker_from_inputs

        generated_files, csv_path = isolate_speaker_from_inputs(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            speaker_name=args.speaker,
            language=args.language,
            target_speaker=args.target_speaker,
            reference_audio=args.reference_audio,
            hf_token=args.hf_token,
            min_segment_seconds=args.min_segment_seconds,
            max_segment_seconds=args.max_segment_seconds,
            drop_overlaps=not args.allow_overlap,
        )
        print(f"Isolated clips: {len(generated_files)}")
        print(f"CSV ready: {csv_path}")
        return


if __name__ == "__main__":
    main()
