import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy
from pydub import AudioSegment

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".flac", ".ogg", ".aac"}


def _iter_supported_files(input_dir: Path) -> List[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def _has_overlap(segment: Tuple[float, float], others: List[Tuple[float, float]]) -> bool:
    start, end = segment
    for other_start, other_end in others:
        if min(end, other_end) - max(start, other_start) > 0:
            return True
    return False


def _choose_speaker_by_duration(segments_by_speaker: Dict[str, List[Tuple[float, float]]]) -> str:
    durations = {
        speaker: sum(end - start for start, end in segments)
        for speaker, segments in segments_by_speaker.items()
    }
    return max(durations, key=durations.get)


def _cosine_similarity(vec_a: numpy.ndarray, vec_b: numpy.ndarray) -> float:
    den = numpy.linalg.norm(vec_a) * numpy.linalg.norm(vec_b)
    if den == 0:
        return -1.0
    return float(numpy.dot(vec_a, vec_b) / den)


def _choose_speaker_by_reference(
    source_file: Path,
    segments_by_speaker: Dict[str, List[Tuple[float, float]]],
    reference_embedding: numpy.ndarray,
    embedding_inference,
    audio_loader,
    min_segment_seconds: float,
    max_segment_seconds: float,
    drop_overlaps: bool,
) -> Optional[str]:
    scores = {}
    for speaker, segments in segments_by_speaker.items():
        other_segments = [segment for label, values in segments_by_speaker.items() if label != speaker for segment in values]
        similarities = []
        for start, end in segments:
            duration = end - start
            if duration < min_segment_seconds or duration > max_segment_seconds:
                continue
            if drop_overlaps and _has_overlap((start, end), other_segments):
                continue

            from pyannote.core import Segment

            waveform, sample_rate = audio_loader.crop(str(source_file), Segment(start, end))
            embedding = embedding_inference({"waveform": waveform, "sample_rate": sample_rate})
            similarities.append(_cosine_similarity(reference_embedding, embedding))

        if similarities:
            scores[speaker] = float(numpy.mean(similarities))

    if not scores:
        return None
    return max(scores, key=scores.get)


def isolate_speaker_from_inputs(
    input_dir: str,
    output_dir: str,
    speaker_name: str,
    language: str = "pt",
    target_speaker: Optional[str] = None,
    reference_audio: Optional[str] = None,
    hf_token: Optional[str] = None,
    min_segment_seconds: float = 1.5,
    max_segment_seconds: float = 15.0,
    drop_overlaps: bool = True,
) -> Tuple[List[str], str]:
    try:
        from pyannote.audio import Audio, Inference, Pipeline
    except Exception as exc:
        raise RuntimeError(
            "pyannote.audio is required for speaker isolation. Install with: pip install pyannote.audio"
        ) from exc

    source_dir = Path(input_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = _iter_supported_files(source_dir)
    if not files:
        raise ValueError(f"No supported audio/video files found in: {input_dir}")

    out_dir = Path(output_dir)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    embedding_inference = Inference("pyannote/embedding", window="whole", use_auth_token=hf_token)
    audio_loader = Audio(sample_rate=16000, mono=True)

    reference_embedding = None
    if reference_audio:
        reference_embedding = embedding_inference(reference_audio)

    rows = []
    generated_files = []

    for source_file in files:
        diarization = pipeline(str(source_file))
        segments_by_speaker: Dict[str, List[Tuple[float, float]]] = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = max(0.0, float(turn.start))
            end = float(turn.end)
            if end <= start:
                continue
            segments_by_speaker.setdefault(speaker, []).append((start, end))

        if not segments_by_speaker:
            continue

        selected_speaker = target_speaker
        if selected_speaker and selected_speaker not in segments_by_speaker:
            raise ValueError(
                f"Speaker '{selected_speaker}' not found in {source_file.name}. "
                f"Detected speakers: {', '.join(sorted(segments_by_speaker))}"
            )

        if not selected_speaker and reference_embedding is not None:
            selected_speaker = _choose_speaker_by_reference(
                source_file=source_file,
                segments_by_speaker=segments_by_speaker,
                reference_embedding=reference_embedding,
                embedding_inference=embedding_inference,
                audio_loader=audio_loader,
                min_segment_seconds=min_segment_seconds,
                max_segment_seconds=max_segment_seconds,
                drop_overlaps=drop_overlaps,
            )

        if not selected_speaker:
            selected_speaker = _choose_speaker_by_duration(segments_by_speaker)

        audio = AudioSegment.from_file(str(source_file))
        other_segments = [
            segment
            for label, values in segments_by_speaker.items()
            if label != selected_speaker
            for segment in values
        ]

        exported = 0
        for start, end in segments_by_speaker[selected_speaker]:
            duration = end - start
            if duration < min_segment_seconds or duration > max_segment_seconds:
                continue
            if drop_overlaps and _has_overlap((start, end), other_segments):
                continue

            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            if end_ms <= start_ms:
                continue

            segment_audio = audio[start_ms:end_ms]
            if len(segment_audio) < int(min_segment_seconds * 1000):
                continue

            output_wav = wav_dir / f"{source_file.stem}_{selected_speaker}_{exported:04d}.wav"
            segment_audio.export(str(output_wav), format="wav")
            exported += 1

            generated_files.append(str(output_wav))
            rows.append([str(output_wav), speaker_name, language])

    if not rows:
        raise RuntimeError("No usable segments produced. Try reducing min_segment_seconds or disabling overlap drop.")

    csv_path = out_dir / "data_from_isolated_speaker.csv"
    with open(csv_path, "w", newline="") as csv_file:
        csv.writer(csv_file).writerows(rows)

    return generated_files, str(csv_path)
