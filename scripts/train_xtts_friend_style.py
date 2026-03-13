#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.app.training import train_gpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Friend-style XTTS fine-tune runner for this project.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--language", default="pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", "--grad-accumm", dest="grad_accum", type=int, default=8)
    parser.add_argument("--max-audio-seconds", type=int, default=11)
    parser.add_argument("--max-text-chars", type=int, default=200)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--restore-path")
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--save-step", type=int, default=1000)
    parser.add_argument("--plot-step", type=int, default=100)
    parser.add_argument("--log-model-step", type=int, default=100)
    parser.add_argument("--batch-group-size", type=int, default=48)
    parser.add_argument("--num-loader-workers", type=int, default=8)
    parser.add_argument("--num-eval-loader-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_gpt(
        language=args.language,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_acumm=args.grad_accum,
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        output_path=args.output_dir,
        max_audio_length=int(args.max_audio_seconds * 22050),
        max_text_length=args.max_text_chars,
        mixed_precision=not args.no_mixed_precision,
        precision=args.precision,
        restore_path=args.restore_path,
        resume_latest=args.resume_latest,
        batch_group_size=args.batch_group_size,
        num_loader_workers=args.num_loader_workers,
        num_eval_loader_workers=args.num_eval_loader_workers,
        save_step=args.save_step,
        plot_step=args.plot_step,
        log_model_step=args.log_model_step,
    )


if __name__ == "__main__":
    main()
