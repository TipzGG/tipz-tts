import argparse

from src.app.audio import enhance_audio_file, process_inputs_folder, process_single_source
from src.app.dataset import build_dataset
from src.app.inference import synthesize_to_file
from src.app.pipeline import run_pipeline
from src.app.profiles import load_voice_registry, preload_voices
from src.app.speaker_isolation import isolate_speaker_from_inputs
from src.app.training import train_gpt


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

    p_train = subparsers.add_parser("train", help="Train XTTS")
    p_train.add_argument("--train-csv", required=True)
    p_train.add_argument("--eval-csv", required=True)
    p_train.add_argument("--language", default="pt")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--grad-accumm", type=int, default=1)
    p_train.add_argument("--max-audio-seconds", type=int, default=11)
    p_train.add_argument("--output-dir", default="out")

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
        result = enhance_audio_file(args.input, args.output)
        print(f"Saved enhanced audio: {result}")
        return

    if args.command == "dataset":
        train_csv, eval_csv = build_dataset(
            base_dataset=args.input_csv,
            output_dir=args.output_dir,
            val_split=args.val_split,
            buffer_seconds=args.buffer_seconds,
            whisper_model_size=args.whisper_model,
            compute_type=args.compute_type,
        )
        print(f"Train metadata: {train_csv}")
        print(f"Eval metadata: {eval_csv}")
        return

    if args.command == "train":
        train_gpt(
            language=args.language,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            grad_acumm=args.grad_accumm,
            train_csv=args.train_csv,
            eval_csv=args.eval_csv,
            output_path=args.output_dir,
            max_audio_length=int(args.max_audio_seconds * 22050),
        )
        return

    if args.command == "infer":
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

        run_pipeline(Path(args.config).resolve())
        return

    if args.command == "import-inputs":
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
