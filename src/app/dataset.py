import csv
import gc
import os
from typing import Iterable, Tuple


def clear_gpu_cache(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def _iter_words(segments: Iterable) -> list:
    words = []
    for segment in segments:
        words.extend(list(segment.words))
    return words


def build_dataset(
    base_dataset: str,
    output_dir: str,
    val_split: float = 0.15,
    buffer_seconds: float = 0.2,
    whisper_model_size: str = "large-v2",
    compute_type: str = "float32",
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
            wav, sample_rate = torchaudio.load(audio_path)

            if wav.size(0) != 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            wav = wav.squeeze()

            segments, _ = asr_model.transcribe(audio_path, word_timestamps=True, language=lang)
            words = _iter_words(list(segments))

            sentence = ""
            sentence_start = 0.0
            first_word = True
            chunk_index = 0

            for word_index, word in enumerate(words):
                if first_word:
                    sentence_start = word.start
                    if word_index == 0:
                        sentence_start = max(sentence_start - buffer_seconds, 0)
                    else:
                        previous_word_end = words[word_index - 1].end
                        sentence_start = max(sentence_start - buffer_seconds, (previous_word_end + sentence_start) / 2)
                    sentence = word.word
                    first_word = False
                else:
                    sentence += word.word

                if not word.word or word.word[-1] not in ["!", ".", "?"]:
                    continue

                cleaned_sentence = multilingual_cleaners(sentence[1:], lang)
                audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))
                relative_audio_file = f"wavs/{audio_file_name}_{str(chunk_index).zfill(8)}.wav"

                if word_index + 1 < len(words):
                    next_word_start = words[word_index + 1].start
                else:
                    next_word_start = (wav.shape[0] - 1) / sample_rate

                word_end = min((word.end + next_word_start) / 2, word.end + buffer_seconds)
                audio = wav[int(sample_rate * sentence_start) : int(sample_rate * word_end)].unsqueeze(0)

                chunk_index += 1
                first_word = True

                if audio.size(-1) < sample_rate / 3:
                    continue

                absolute_audio_file = os.path.join(output_dir, relative_audio_file)
                os.makedirs(os.path.dirname(absolute_audio_file), exist_ok=True)
                torchaudio.save(absolute_audio_file, audio, sample_rate)

                metadata["audio_file"].append(relative_audio_file)
                metadata["text"].append(cleaned_sentence)
                metadata["speaker_name"].append(speaker_name)

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
