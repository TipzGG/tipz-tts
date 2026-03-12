import csv
import gc
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple


def clear_gpu_cache(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def _iter_words(segments: Iterable) -> list:
    words = []
    for segment in segments:
        words.extend(list(segment.words))
    return words


def _current_text(parts: list[str]) -> str:
    return "".join(parts).strip()


def _should_flush_chunk(
    text: str,
    sentence_start: float,
    current_word_end: float,
    max_text_chars: int,
    max_segment_seconds: float,
) -> bool:
    if not text:
        return False
    if text[-1] in ["!", ".", "?"]:
        return True
    if len(text) >= max_text_chars:
        return True
    if current_word_end - sentence_start >= max_segment_seconds:
        return True
    return False


def _would_exceed_limits(
    current_text: str,
    next_word: str,
    sentence_start: float,
    next_word_end: float,
    max_text_chars: int,
    max_segment_seconds: float,
) -> bool:
    candidate = f"{current_text}{next_word}".strip()
    if not current_text:
        return False
    if len(candidate) > max_text_chars:
        return True
    if next_word_end - sentence_start > max_segment_seconds:
        return True
    return False


def _resolve_audio_path(metadata_csv: str, audio_file: str) -> str:
    return str((Path(metadata_csv).resolve().parent / audio_file).resolve())


def _merge_timing_ranges(ranges: list[tuple[int, int]], gap_ms: int) -> list[tuple[int, int]]:
    if not ranges:
        return []

    merged: list[tuple[int, int]] = []
    current_start, current_end = ranges[0]
    for start, end in ranges[1:]:
        if start - current_end <= gap_ms:
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def _split_range_by_max_length(start_ms: int, end_ms: int, max_chunk_ms: int) -> list[tuple[int, int]]:
    if end_ms <= start_ms:
        return []

    if end_ms - start_ms <= max_chunk_ms:
        return [(start_ms, end_ms)]

    chunks: list[tuple[int, int]] = []
    cursor = start_ms
    while cursor < end_ms:
        next_end = min(cursor + max_chunk_ms, end_ms)
        chunks.append((cursor, next_end))
        cursor = next_end
    return chunks


def _plan_pre_asr_chunks(
    nonsilent_ranges_ms: list[tuple[int, int]],
    *,
    audio_duration_ms: int,
    max_chunk_seconds: float,
    min_chunk_seconds: float,
    keep_silence_ms: int,
    merge_gap_ms: int,
) -> list[tuple[int, int]]:
    max_chunk_ms = int(max_chunk_seconds * 1000)
    min_chunk_ms = int(min_chunk_seconds * 1000)
    if max_chunk_ms <= 0:
        raise ValueError("max_chunk_seconds must be > 0")
    if min_chunk_ms < 0:
        raise ValueError("min_chunk_seconds must be >= 0")

    if not nonsilent_ranges_ms:
        return [(0, audio_duration_ms)]

    expanded = [
        (max(0, start - keep_silence_ms), min(audio_duration_ms, end + keep_silence_ms))
        for start, end in nonsilent_ranges_ms
        if end > start
    ]
    merged = _merge_timing_ranges(expanded, gap_ms=merge_gap_ms)

    planned: list[tuple[int, int]] = []
    for start, end in merged:
        planned.extend(_split_range_by_max_length(start, end, max_chunk_ms=max_chunk_ms))

    filtered = [(start, end) for start, end in planned if end - start >= min_chunk_ms]
    return filtered or [(0, audio_duration_ms)]


def _prepare_pre_asr_chunks(
    audio_path: str,
    chunks_dir: str,
    *,
    max_chunk_seconds: float,
    min_chunk_seconds: float,
    min_silence_len_ms: int,
    silence_thresh_db: int,
    keep_silence_ms: int,
    merge_gap_ms: int,
) -> list[str]:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    audio = AudioSegment.from_wav(audio_path)
    duration_ms = len(audio)
    nonsilent_ranges = [
        (int(start), int(end))
        for start, end in detect_nonsilent(
            audio,
            min_silence_len=min_silence_len_ms,
            silence_thresh=silence_thresh_db,
        )
    ]
    planned_ranges = _plan_pre_asr_chunks(
        nonsilent_ranges,
        audio_duration_ms=duration_ms,
        max_chunk_seconds=max_chunk_seconds,
        min_chunk_seconds=min_chunk_seconds,
        keep_silence_ms=keep_silence_ms,
        merge_gap_ms=merge_gap_ms,
    )

    chunks_path = Path(chunks_dir)
    chunks_path.mkdir(parents=True, exist_ok=True)
    source_stem = Path(audio_path).stem
    chunk_files: list[str] = []
    for index, (start_ms, end_ms) in enumerate(planned_ranges):
        chunk_audio = audio[start_ms:end_ms]
        chunk_path = chunks_path / f"{source_stem}_chunk_{index:04d}.wav"
        chunk_audio.export(chunk_path, format="wav")
        chunk_files.append(str(chunk_path))
    return chunk_files


def _safe_audio_duration_seconds(audio_path: str) -> float:
    import soundfile as sf

    info = sf.info(audio_path)
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def _text_has_suspicious_chars(text: str) -> bool:
    allowed_chars = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        " .,;:!?-'\"/()%"
        "áàâãéêíóôõúüç"
        "ÁÀÂÃÉÊÍÓÔÕÚÜÇ"
    )
    return any(char not in allowed_chars for char in text)


def _flag_penalty(flag: str) -> int:
    penalties = {
        "audio_too_short": 25,
        "audio_too_long": 20,
        "text_too_short": 10,
        "text_too_long": 15,
        "chars_per_second_low": 15,
        "chars_per_second_high": 15,
        "ellipsis": 5,
        "many_commas": 10,
        "filler_words": 10,
        "suspicious_chars": 10,
        "double_spaces": 5,
        "no_terminal_punctuation": 5,
        "rare_word_ratio_high": 20,
        "rare_long_word_count_high": 15,
    }
    return penalties.get(flag, 0)


def _status_threshold_for_policy(policy: str) -> int:
    normalized = policy.strip().lower()
    thresholds = {
        "strict": 75,
        "balanced": 65,
        "aggressive": 55,
    }
    if normalized not in thresholds:
        raise ValueError(f"Unknown auto-curation policy: {policy}")
    return thresholds[normalized]


def _score_flags(flags: list[str]) -> int:
    score = 100
    for flag in flags:
        score -= _flag_penalty(flag)
    return max(0, score)


def _normalize_token(text: str) -> str:
    return text.strip(" \t\r\n.,;:!?-_'\"/()[]{}").lower()


def _token_frequencies(texts: list[str]) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for text in texts:
        for token in (_normalize_token(part) for part in text.split()):
            if len(token) < 3:
                continue
            frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies


def _dataset_level_text_flags(text: str, token_frequencies: dict[str, int]) -> list[str]:
    tokens = [_normalize_token(part) for part in text.split()]
    informative_tokens = [token for token in tokens if len(token) >= 5]
    if not informative_tokens:
        return []

    rare_tokens = [token for token in informative_tokens if token_frequencies.get(token, 0) <= 1]
    rare_ratio = len(rare_tokens) / len(informative_tokens)

    flags: list[str] = []
    if len(rare_tokens) >= 2 and rare_ratio >= 0.34:
        flags.append("rare_word_ratio_high")
    if len(rare_tokens) >= 4:
        flags.append("rare_long_word_count_high")
    return flags


def _review_row(
    row: dict,
    metadata_csv: str,
    *,
    min_audio_seconds: float,
    max_audio_seconds: float,
    min_text_chars: int,
    max_text_chars: int,
    min_chars_per_second: float,
    max_chars_per_second: float,
) -> dict:
    text = str(row.get("text", "")).strip()
    audio_file = str(row.get("audio_file", "")).strip()
    audio_path = _resolve_audio_path(metadata_csv, audio_file)
    duration_seconds = _safe_audio_duration_seconds(audio_path)
    text_chars = len(text)
    word_count = len(text.split())
    chars_per_second = round(text_chars / duration_seconds, 2) if duration_seconds > 0 else 0.0

    flags: list[str] = []
    if duration_seconds < min_audio_seconds:
        flags.append("audio_too_short")
    if duration_seconds > max_audio_seconds:
        flags.append("audio_too_long")
    if text_chars < min_text_chars:
        flags.append("text_too_short")
    if text_chars > max_text_chars:
        flags.append("text_too_long")
    if chars_per_second < min_chars_per_second:
        flags.append("chars_per_second_low")
    if chars_per_second > max_chars_per_second:
        flags.append("chars_per_second_high")
    if text.count("...") or "…" in text:
        flags.append("ellipsis")
    if text.count(",") >= 4:
        flags.append("many_commas")
    if any(token in text.lower() for token in [" ah ", " eh ", " hum ", " uh "]):
        flags.append("filler_words")
    if _text_has_suspicious_chars(text):
        flags.append("suspicious_chars")
    if "  " in text:
        flags.append("double_spaces")
    if not text.endswith((".", "!", "?")):
        flags.append("no_terminal_punctuation")
    reviewed = dict(row)
    reviewed["audio_path"] = audio_path
    reviewed["duration_seconds"] = round(duration_seconds, 3)
    reviewed["text_chars"] = text_chars
    reviewed["word_count"] = word_count
    reviewed["chars_per_second"] = chars_per_second
    reviewed["flags"] = ",".join(flags)
    reviewed["review_status"] = "review" if flags else "keep"
    return reviewed


def review_dataset(
    metadata_csv: str,
    output_csv: Optional[str] = None,
    filtered_output_csv: Optional[str] = None,
    *,
    min_audio_seconds: float = 1.0,
    max_audio_seconds: float = 12.0,
    min_text_chars: int = 8,
    max_text_chars: int = 180,
    min_chars_per_second: float = 4.0,
    max_chars_per_second: float = 24.0,
    auto_status_policy: str = "strict",
) -> Tuple[str, Optional[str]]:
    import pandas

    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"{metadata_csv} not found")

    dataframe = pandas.read_csv(metadata_csv, sep="|")
    if dataframe.empty:
        raise RuntimeError("Metadata CSV is empty.")

    reviewed_rows = [
        _review_row(
            row,
            metadata_csv,
            min_audio_seconds=min_audio_seconds,
            max_audio_seconds=max_audio_seconds,
            min_text_chars=min_text_chars,
            max_text_chars=max_text_chars,
            min_chars_per_second=min_chars_per_second,
            max_chars_per_second=max_chars_per_second,
        )
        for row in dataframe.to_dict(orient="records")
    ]
    token_frequencies = _token_frequencies([str(row.get("text", "")) for row in reviewed_rows])
    for row in reviewed_rows:
        flags = [flag for flag in str(row.get("flags", "")).split(",") if flag]
        flags.extend(_dataset_level_text_flags(str(row.get("text", "")), token_frequencies))
        deduped_flags = list(dict.fromkeys(flags))
        row["flags"] = ",".join(deduped_flags)
        row["score"] = _score_flags(deduped_flags)
        row["review_status"] = "review" if deduped_flags else "keep"

    reviewed_dataframe = pandas.DataFrame(reviewed_rows).sort_values(["review_status", "audio_file"])
    score_threshold = _status_threshold_for_policy(auto_status_policy)
    reviewed_dataframe["auto_status"] = reviewed_dataframe["score"].apply(
        lambda score: "keep" if int(score) >= score_threshold else "drop"
    )
    reviewed_dataframe["auto_status_policy"] = auto_status_policy

    if output_csv is None:
        output_csv = str(Path(metadata_csv).with_name(f"{Path(metadata_csv).stem}_review.csv"))
    reviewed_dataframe.to_csv(output_csv, sep="|", index=False)

    if filtered_output_csv:
        filtered_dataframe = reviewed_dataframe[reviewed_dataframe["auto_status"] == "keep"][
            ["audio_file", "text", "speaker_name"]
        ]
        filtered_dataframe.to_csv(filtered_output_csv, sep="|", index=False)

    return output_csv, filtered_output_csv


def auto_curate_dataset_splits(
    train_csv: str,
    eval_csv: str,
    output_dir: str,
    *,
    policy: str = "strict",
    min_audio_seconds: float = 1.0,
    max_audio_seconds: float = 12.0,
    min_text_chars: int = 8,
    max_text_chars: int = 180,
    min_chars_per_second: float = 4.0,
    max_chars_per_second: float = 24.0,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_scored_csv = str(output_path / "metadata_train_scored.csv")
    eval_scored_csv = str(output_path / "metadata_eval_scored.csv")
    train_auto_csv = str(output_path / "metadata_train_auto.csv")
    eval_auto_csv = str(output_path / "metadata_eval_auto.csv")

    review_dataset(
        metadata_csv=train_csv,
        output_csv=train_scored_csv,
        filtered_output_csv=train_auto_csv,
        min_audio_seconds=min_audio_seconds,
        max_audio_seconds=max_audio_seconds,
        min_text_chars=min_text_chars,
        max_text_chars=max_text_chars,
        min_chars_per_second=min_chars_per_second,
        max_chars_per_second=max_chars_per_second,
        auto_status_policy=policy,
    )
    review_dataset(
        metadata_csv=eval_csv,
        output_csv=eval_scored_csv,
        filtered_output_csv=eval_auto_csv,
        min_audio_seconds=min_audio_seconds,
        max_audio_seconds=max_audio_seconds,
        min_text_chars=min_text_chars,
        max_text_chars=max_text_chars,
        min_chars_per_second=min_chars_per_second,
        max_chars_per_second=max_chars_per_second,
        auto_status_policy=policy,
    )

    return {
        "train_scored_csv": train_scored_csv,
        "eval_scored_csv": eval_scored_csv,
        "train_auto_csv": train_auto_csv,
        "eval_auto_csv": eval_auto_csv,
        "policy": policy,
    }


def _finalize_chunk(
    *,
    sentence_parts: list[str],
    sentence_start: float,
    chunk_end: float,
    next_word_start: float,
    buffer_seconds: float,
    lang: str,
    wav,
    sample_rate: int,
    output_dir: str,
    audio_file_name: str,
    chunk_index: int,
    speaker_name: str,
    metadata: dict,
    torchaudio_module,
    multilingual_cleaners,
) -> Optional[int]:
    sentence_text = _current_text(sentence_parts)
    if not sentence_text:
        return None

    cleaned_sentence = multilingual_cleaners(sentence_text, lang).strip()
    if not cleaned_sentence:
        return None

    relative_audio_file = f"wavs/{audio_file_name}_{str(chunk_index).zfill(8)}.wav"
    word_end = min((chunk_end + next_word_start) / 2, chunk_end + buffer_seconds)
    audio = wav[int(sample_rate * sentence_start) : int(sample_rate * word_end)].unsqueeze(0)

    if audio.size(-1) < sample_rate / 3:
        return None

    absolute_audio_file = os.path.join(output_dir, relative_audio_file)
    os.makedirs(os.path.dirname(absolute_audio_file), exist_ok=True)
    torchaudio_module.save(absolute_audio_file, audio, sample_rate)

    metadata["audio_file"].append(relative_audio_file)
    metadata["text"].append(cleaned_sentence)
    metadata["speaker_name"].append(speaker_name)
    return chunk_index + 1


def build_dataset(
    base_dataset: str,
    output_dir: str,
    val_split: float = 0.15,
    buffer_seconds: float = 0.2,
    whisper_model_size: str = "large-v2",
    compute_type: str = "float32",
    max_segment_seconds: float = 11.0,
    max_text_chars: int = 200,
    pre_asr_max_chunk_seconds: float = 45.0,
    pre_asr_min_chunk_seconds: float = 2.0,
    pre_asr_min_silence_ms: int = 700,
    pre_asr_keep_silence_ms: int = 200,
    pre_asr_merge_gap_ms: int = 250,
    pre_asr_silence_thresh_db: int = -40,
) -> Tuple[str, str]:
    import pandas
    import torch
    import torchaudio
    from faster_whisper import WhisperModel
    from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners

    torch.set_num_threads(16)

    if not os.path.exists(base_dataset):
        raise FileNotFoundError(f"{base_dataset} not found")

    os.makedirs(output_dir, exist_ok=True)
    pre_asr_chunks_dir = os.path.join(output_dir, "_pre_asr_chunks")
    os.makedirs(pre_asr_chunks_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)

    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    with open(base_dataset, mode="r", newline="") as archive:
        reader = csv.reader(archive, delimiter=",")
        for line_number, line in enumerate(reader):
            if len(line) != 3:
                print(f"Skipping invalid line {line_number}: {line}")
                continue

            audio_path, speaker_name, lang = [item.strip() for item in line]
            pre_asr_chunk_files = _prepare_pre_asr_chunks(
                audio_path=audio_path,
                chunks_dir=os.path.join(pre_asr_chunks_dir, Path(audio_path).stem),
                max_chunk_seconds=pre_asr_max_chunk_seconds,
                min_chunk_seconds=pre_asr_min_chunk_seconds,
                min_silence_len_ms=pre_asr_min_silence_ms,
                silence_thresh_db=pre_asr_silence_thresh_db,
                keep_silence_ms=pre_asr_keep_silence_ms,
                merge_gap_ms=pre_asr_merge_gap_ms,
            )

            chunk_index = 0
            audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))
            for pre_asr_chunk_file in pre_asr_chunk_files:
                wav, sample_rate = torchaudio.load(pre_asr_chunk_file)

                if wav.size(0) != 1:
                    wav = torch.mean(wav, dim=0, keepdim=True)
                wav = wav.squeeze()

                segments, _ = asr_model.transcribe(pre_asr_chunk_file, word_timestamps=True, language=lang)
                words = _iter_words(list(segments))
                if not words:
                    continue

                sentence_parts = []
                sentence_start = 0.0
                first_word = True

                for word_index, word in enumerate(words):
                    if not first_word and _would_exceed_limits(
                        current_text=_current_text(sentence_parts),
                        next_word=word.word,
                        sentence_start=sentence_start,
                        next_word_end=word.end,
                        max_text_chars=max_text_chars,
                        max_segment_seconds=max_segment_seconds,
                    ):
                        next_chunk_index = _finalize_chunk(
                            sentence_parts=sentence_parts,
                            sentence_start=sentence_start,
                            chunk_end=words[word_index - 1].end,
                            next_word_start=word.start,
                            buffer_seconds=buffer_seconds,
                            lang=lang,
                            wav=wav,
                            sample_rate=sample_rate,
                            output_dir=output_dir,
                            audio_file_name=audio_file_name,
                            chunk_index=chunk_index,
                            speaker_name=speaker_name,
                            metadata=metadata,
                            torchaudio_module=torchaudio,
                            multilingual_cleaners=multilingual_cleaners,
                        )
                        if next_chunk_index is not None:
                            chunk_index = next_chunk_index
                        first_word = True
                        sentence_parts = []

                    if first_word:
                        sentence_start = word.start
                        if word_index == 0:
                            sentence_start = max(sentence_start - buffer_seconds, 0)
                        else:
                            previous_word_end = words[word_index - 1].end
                            sentence_start = max(
                                sentence_start - buffer_seconds,
                                (previous_word_end + sentence_start) / 2,
                            )
                        sentence_parts = [word.word]
                        first_word = False
                    else:
                        sentence_parts.append(word.word)

                    current_text = _current_text(sentence_parts)
                    if not _should_flush_chunk(
                        text=current_text,
                        sentence_start=sentence_start,
                        current_word_end=word.end,
                        max_text_chars=max_text_chars,
                        max_segment_seconds=max_segment_seconds,
                    ):
                        continue

                    if word_index + 1 < len(words):
                        next_word_start = words[word_index + 1].start
                    else:
                        next_word_start = (wav.shape[0] - 1) / sample_rate

                    next_chunk_index = _finalize_chunk(
                        sentence_parts=sentence_parts,
                        sentence_start=sentence_start,
                        chunk_end=word.end,
                        next_word_start=next_word_start,
                        buffer_seconds=buffer_seconds,
                        lang=lang,
                        wav=wav,
                        sample_rate=sample_rate,
                        output_dir=output_dir,
                        audio_file_name=audio_file_name,
                        chunk_index=chunk_index,
                        speaker_name=speaker_name,
                        metadata=metadata,
                        torchaudio_module=torchaudio,
                        multilingual_cleaners=multilingual_cleaners,
                    )
                    if next_chunk_index is not None:
                        chunk_index = next_chunk_index
                    first_word = True
                    sentence_parts = []

                if sentence_parts:
                    next_chunk_index = _finalize_chunk(
                        sentence_parts=sentence_parts,
                        sentence_start=sentence_start,
                        chunk_end=words[-1].end,
                        next_word_start=(wav.shape[0] - 1) / sample_rate,
                        buffer_seconds=buffer_seconds,
                        lang=lang,
                        wav=wav,
                        sample_rate=sample_rate,
                        output_dir=output_dir,
                        audio_file_name=audio_file_name,
                        chunk_index=chunk_index,
                        speaker_name=speaker_name,
                        metadata=metadata,
                        torchaudio_module=torchaudio,
                        multilingual_cleaners=multilingual_cleaners,
                    )
                    if next_chunk_index is not None:
                        chunk_index = next_chunk_index

            clear_gpu_cache(torch)

    dataframe = pandas.DataFrame(metadata)
    if dataframe.empty:
        raise RuntimeError("No valid audio segments were produced. Check your source files.")

    dataframe = dataframe.sample(frac=1)
    val_size = int(len(dataframe) * val_split)

    eval_dataframe = dataframe[:val_size].sort_values("audio_file")
    train_dataframe = dataframe[val_size:].sort_values("audio_file")

    train_metadata_path = os.path.join(output_dir, "metadata_train.csv")
    eval_metadata_path = os.path.join(output_dir, "metadata_eval.csv")

    train_dataframe.to_csv(train_metadata_path, sep="|", index=False)
    eval_dataframe.to_csv(eval_metadata_path, sep="|", index=False)

    del asr_model, train_dataframe, eval_dataframe, dataframe, metadata
    gc.collect()

    return train_metadata_path, eval_metadata_path
