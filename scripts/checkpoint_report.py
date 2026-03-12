#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CHECKPOINT_PATTERN = re.compile(r"checkpoint_(\d+)\.pth$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report and rank XTTS checkpoints.")
    parser.add_argument(
        "--training-dir",
        default="outputs/beerschool/training_test/run/training",
        help="Directory that contains GPT_XTTS_FT-* run folders or a specific run folder.",
    )
    parser.add_argument(
        "--run-dir",
        help="Specific run directory (overrides --training-dir auto discovery).",
    )
    return parser.parse_args()


def _checkpoint_step(path: Path) -> Optional[int]:
    match = CHECKPOINT_PATTERN.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def _pick_run_dir(training_dir: Path, run_dir: Optional[str]) -> Path:
    if run_dir:
        selected = Path(run_dir).resolve()
        if not selected.exists():
            raise FileNotFoundError(f"Run directory not found: {selected}")
        return selected

    root = training_dir.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Training directory not found: {root}")

    candidates = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("GPT_XTTS_FT-")]
    if not candidates and root.name.startswith("GPT_XTTS_FT-"):
        return root
    if not candidates:
        raise RuntimeError(f"No run folders found inside: {root}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def _list_checkpoints(run_path: Path) -> List[Tuple[int, Path]]:
    checkpoints = []
    for path in run_path.glob("checkpoint_*.pth"):
        step = _checkpoint_step(path)
        if step is None:
            continue
        checkpoints.append((step, path.resolve()))
    return sorted(checkpoints, key=lambda item: item[0])


def _select_loss_tag(tags: List[str]) -> Optional[str]:
    if not tags:
        return None
    preferred = [
        "eval/loss",
        "val/loss",
        "validation/loss",
        "loss/eval",
        "train/loss",
        "loss",
    ]
    normalized = {tag.lower(): tag for tag in tags}
    for candidate in preferred:
        if candidate in normalized:
            return normalized[candidate]
    loss_tags = [tag for tag in tags if "loss" in tag.lower()]
    if not loss_tags:
        return None
    return sorted(loss_tags)[0]


def _load_loss_by_step(run_path: Path) -> Tuple[Optional[str], Dict[int, float]]:
    event_files = sorted(run_path.glob("events.out.tfevents.*"))
    if not event_files:
        return None, {}

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return None, {}

    all_tags = []
    values: Dict[str, Dict[int, float]] = {}
    for event_path in event_files:
        accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception:
            continue
        scalar_tags = accumulator.Tags().get("scalars", [])
        all_tags.extend(scalar_tags)
        for tag in scalar_tags:
            tag_values = values.setdefault(tag, {})
            for scalar in accumulator.Scalars(tag):
                tag_values[int(scalar.step)] = float(scalar.value)

    selected_tag = _select_loss_tag(sorted(set(all_tags)))
    if not selected_tag:
        return None, {}
    return selected_tag, values.get(selected_tag, {})


def _nearest_loss_at_or_before_step(step: int, loss_by_step: Dict[int, float]) -> Optional[float]:
    available_steps = [candidate for candidate in loss_by_step.keys() if candidate <= step]
    if not available_steps:
        return None
    nearest = max(available_steps)
    return loss_by_step[nearest]


def main() -> None:
    args = parse_args()
    run_path = _pick_run_dir(Path(args.training_dir), args.run_dir)
    checkpoints = _list_checkpoints(run_path)
    if not checkpoints:
        raise RuntimeError(f"No checkpoint_*.pth files found in: {run_path}")

    tag_name, loss_by_step = _load_loss_by_step(run_path)
    scored_rows = []
    for step, checkpoint_path in checkpoints:
        scored_rows.append(
            {
                "step": step,
                "checkpoint": checkpoint_path,
                "loss": _nearest_loss_at_or_before_step(step, loss_by_step),
            }
        )

    print(f"run_dir: {run_path}")
    print(f"checkpoints_found: {len(scored_rows)}")
    print()
    print("Checkpoint ranking")
    print("==================")
    if tag_name and loss_by_step:
        print(f"loss_tag: {tag_name}")
        ranked = sorted(
            scored_rows,
            key=lambda row: (row["loss"] is None, row["loss"] if row["loss"] is not None else float("inf")),
        )
        for index, row in enumerate(ranked, start=1):
            loss_text = f"{row['loss']:.6f}" if row["loss"] is not None else "n/a"
            print(f"{index:>2}. step={row['step']:<6} loss={loss_text:<12} path={row['checkpoint']}")
        best = ranked[0]
        print()
        print(f"suggested_checkpoint: {best['checkpoint']}")
        print("selection_rule: lowest available loss near checkpoint step")
    else:
        print("No TensorBoard loss scalars found. Falling back to latest checkpoint.")
        latest = scored_rows[-1]
        for index, row in enumerate(scored_rows, start=1):
            print(f"{index:>2}. step={row['step']:<6} path={row['checkpoint']}")
        print()
        print(f"suggested_checkpoint: {latest['checkpoint']}")
        print("selection_rule: highest checkpoint step (latest)")


if __name__ == "__main__":
    main()
