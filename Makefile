PYTHON ?= python3.11
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

VOICE ?= sample_voice
TTS_LANGUAGE ?= pt
WORKSPACE ?= outputs/$(VOICE)
INPUT_DIR ?= inputs/$(VOICE)
RAW_DIR ?= $(WORKSPACE)/raw
ISOLATED_DIR ?= $(WORKSPACE)/isolated
DATASET_DIR ?= $(WORKSPACE)/dataset
SOURCE_CSV ?= $(RAW_DIR)/data_from_inputs.csv
TRAIN_CSV ?= $(DATASET_DIR)/metadata_train.csv
EVAL_CSV ?= $(DATASET_DIR)/metadata_eval.csv
REVIEW_CSV ?= $(DATASET_DIR)/metadata_train_review.csv
CURATED_CSV ?= $(DATASET_DIR)/metadata_train_curated.csv

.PHONY: \
	venv install install-local install-speaker-isolation fix-venv test \
	pipeline pipeline-youtube-auto print-pipeline-vars import-inputs isolate-speaker build-dataset review-dataset \
	pipeline-local pipeline-isolated train train-curated infer download enhance serve clean

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements-dev.txt

install-local:
	./install.sh

install-speaker-isolation:
	$(PIP) install -r requirements-speaker-isolation.txt

fix-venv:
	@if [ ! -x "$(PY)" ]; then echo "Virtualenv not found. Run: make install-local"; exit 1; fi
	$(PY) -m pip uninstall -y coqpit >/dev/null 2>&1 || true
	$(PY) -c "import shutil, site; from pathlib import Path; [shutil.rmtree(Path(base) / 'coqpit', ignore_errors=True) for base in site.getsitepackages()]"
	$(PIP) install --force-reinstall --no-deps coqpit-config==0.2.4

test:
	$(PY) -m unittest discover -s tests -p 'test_*.py'

pipeline:
	@if [ -z "$(CONFIG)" ]; then echo "Use: make pipeline CONFIG=templates/minha_voz.json"; exit 1; fi
	$(PY) cli.py pipeline --config $(CONFIG)

pipeline-youtube-auto:
	@if [ -z "$(CONFIG)" ]; then echo "Use: make pipeline-youtube-auto CONFIG=templates/minha_voz.json"; exit 1; fi
	$(PY) cli.py pipeline --config $(CONFIG)

print-pipeline-vars:
	@echo "VOICE=$(VOICE)"
	@echo "TTS_LANGUAGE=$(TTS_LANGUAGE)"
	@echo "INPUT_DIR=$(INPUT_DIR)"
	@echo "WORKSPACE=$(WORKSPACE)"
	@echo "RAW_DIR=$(RAW_DIR)"
	@echo "ISOLATED_DIR=$(ISOLATED_DIR)"
	@echo "DATASET_DIR=$(DATASET_DIR)"
	@echo "SOURCE_CSV=$(SOURCE_CSV)"
	@echo "TRAIN_CSV=$(TRAIN_CSV)"
	@echo "EVAL_CSV=$(EVAL_CSV)"
	@echo "REVIEW_CSV=$(REVIEW_CSV)"
	@echo "CURATED_CSV=$(CURATED_CSV)"

import-inputs:
	@if [ -z "$(VOICE)" ]; then echo "Use: make import-inputs VOICE=minha_voz INPUT_DIR=inputs/minha_voz"; exit 1; fi
	$(PY) cli.py import-inputs --input-dir $(INPUT_DIR) --output-folder $(RAW_DIR) --speaker $(VOICE) --language $(TTS_LANGUAGE) $(if $(TRIM_START),--trim-start $(TRIM_START),) $(if $(TRIM_END),--trim-end $(TRIM_END),) $(if $(SKIP_ENHANCEMENT),--skip-enhancement,)

isolate-speaker:
	@if [ -z "$(VOICE)" ]; then echo "Use: make isolate-speaker VOICE=minha_voz INPUT_DIR=inputs/minha_voz REFERENCE_AUDIO=inputs/ref.wav"; exit 1; fi
	$(PY) cli.py isolate-speaker --input-dir $(INPUT_DIR) --output-dir $(ISOLATED_DIR) --speaker $(VOICE) --language $(TTS_LANGUAGE) $(if $(TARGET_SPEAKER),--target-speaker $(TARGET_SPEAKER),) $(if $(REFERENCE_AUDIO),--reference-audio $(REFERENCE_AUDIO),) $(if $(HF_TOKEN),--hf-token $(HF_TOKEN),) $(if $(MIN_SEGMENT_SECONDS),--min-segment-seconds $(MIN_SEGMENT_SECONDS),) $(if $(MAX_SEGMENT_SECONDS),--max-segment-seconds $(MAX_SEGMENT_SECONDS),) $(if $(ALLOW_OVERLAP),--allow-overlap,)

build-dataset:
	@if [ -z "$(SOURCE_CSV)" ]; then echo "Use: make build-dataset SOURCE_CSV=outputs/minha_voz/raw/data_from_inputs.csv"; exit 1; fi
	$(PY) cli.py dataset --input-csv $(SOURCE_CSV) --output-dir $(DATASET_DIR) --val-split $(or $(VAL_SPLIT),0.15) --buffer-seconds $(or $(BUFFER_SECONDS),0.2) --whisper-model $(or $(WHISPER_MODEL),large-v2) --compute-type $(or $(COMPUTE_TYPE),float32) --max-segment-seconds $(or $(MAX_SEGMENT_SECONDS_DATASET),11.0) --max-text-chars $(or $(MAX_TEXT_CHARS),200) --pre-asr-max-chunk-seconds $(or $(PRE_ASR_MAX_CHUNK_SECONDS),45.0) --pre-asr-min-chunk-seconds $(or $(PRE_ASR_MIN_CHUNK_SECONDS),2.0) --pre-asr-min-silence-ms $(or $(PRE_ASR_MIN_SILENCE_MS),700) --pre-asr-keep-silence-ms $(or $(PRE_ASR_KEEP_SILENCE_MS),200) --pre-asr-merge-gap-ms $(or $(PRE_ASR_MERGE_GAP_MS),250) --pre-asr-silence-thresh-db $(or $(PRE_ASR_SILENCE_THRESH_DB),-40)

review-dataset:
	@if [ ! -f "$(TRAIN_CSV)" ]; then echo "Train CSV not found: $(TRAIN_CSV)"; exit 1; fi
	$(PY) cli.py review-dataset --metadata-csv $(TRAIN_CSV) --output-csv $(REVIEW_CSV) --filtered-output-csv $(CURATED_CSV) --min-audio-seconds $(or $(MIN_AUDIO_SECONDS),1.0) --max-audio-seconds $(or $(MAX_AUDIO_SECONDS_REVIEW),12.0) --min-text-chars $(or $(MIN_TEXT_CHARS),8) --max-text-chars $(or $(MAX_TEXT_CHARS_REVIEW),180) --min-chars-per-second $(or $(MIN_CHARS_PER_SECOND),4.0) --max-chars-per-second $(or $(MAX_CHARS_PER_SECOND),24.0)

pipeline-local:
	$(MAKE) import-inputs VOICE="$(VOICE)" TTS_LANGUAGE="$(TTS_LANGUAGE)" INPUT_DIR="$(INPUT_DIR)" WORKSPACE="$(WORKSPACE)" RAW_DIR="$(RAW_DIR)" TRIM_START="$(TRIM_START)" TRIM_END="$(TRIM_END)" SKIP_ENHANCEMENT="$(SKIP_ENHANCEMENT)"
	$(MAKE) build-dataset VOICE="$(VOICE)" TTS_LANGUAGE="$(TTS_LANGUAGE)" WORKSPACE="$(WORKSPACE)" DATASET_DIR="$(DATASET_DIR)" SOURCE_CSV="$(RAW_DIR)/data_from_inputs.csv" VAL_SPLIT="$(VAL_SPLIT)" BUFFER_SECONDS="$(BUFFER_SECONDS)" WHISPER_MODEL="$(WHISPER_MODEL)" COMPUTE_TYPE="$(COMPUTE_TYPE)" MAX_SEGMENT_SECONDS_DATASET="$(MAX_SEGMENT_SECONDS_DATASET)" MAX_TEXT_CHARS="$(MAX_TEXT_CHARS)" PRE_ASR_MAX_CHUNK_SECONDS="$(PRE_ASR_MAX_CHUNK_SECONDS)" PRE_ASR_MIN_CHUNK_SECONDS="$(PRE_ASR_MIN_CHUNK_SECONDS)" PRE_ASR_MIN_SILENCE_MS="$(PRE_ASR_MIN_SILENCE_MS)" PRE_ASR_KEEP_SILENCE_MS="$(PRE_ASR_KEEP_SILENCE_MS)" PRE_ASR_MERGE_GAP_MS="$(PRE_ASR_MERGE_GAP_MS)" PRE_ASR_SILENCE_THRESH_DB="$(PRE_ASR_SILENCE_THRESH_DB)"
	$(MAKE) review-dataset VOICE="$(VOICE)" WORKSPACE="$(WORKSPACE)" DATASET_DIR="$(DATASET_DIR)" TRAIN_CSV="$(DATASET_DIR)/metadata_train.csv" REVIEW_CSV="$(REVIEW_CSV)" CURATED_CSV="$(CURATED_CSV)" MIN_AUDIO_SECONDS="$(MIN_AUDIO_SECONDS)" MAX_AUDIO_SECONDS_REVIEW="$(MAX_AUDIO_SECONDS_REVIEW)" MIN_TEXT_CHARS="$(MIN_TEXT_CHARS)" MAX_TEXT_CHARS_REVIEW="$(MAX_TEXT_CHARS_REVIEW)" MIN_CHARS_PER_SECOND="$(MIN_CHARS_PER_SECOND)" MAX_CHARS_PER_SECOND="$(MAX_CHARS_PER_SECOND)"

pipeline-isolated:
	$(MAKE) isolate-speaker VOICE="$(VOICE)" TTS_LANGUAGE="$(TTS_LANGUAGE)" INPUT_DIR="$(INPUT_DIR)" WORKSPACE="$(WORKSPACE)" ISOLATED_DIR="$(ISOLATED_DIR)" TARGET_SPEAKER="$(TARGET_SPEAKER)" REFERENCE_AUDIO="$(REFERENCE_AUDIO)" HF_TOKEN="$(HF_TOKEN)" MIN_SEGMENT_SECONDS="$(MIN_SEGMENT_SECONDS)" MAX_SEGMENT_SECONDS="$(MAX_SEGMENT_SECONDS)" ALLOW_OVERLAP="$(ALLOW_OVERLAP)"
	$(MAKE) build-dataset VOICE="$(VOICE)" TTS_LANGUAGE="$(TTS_LANGUAGE)" WORKSPACE="$(WORKSPACE)" DATASET_DIR="$(DATASET_DIR)" SOURCE_CSV="$(ISOLATED_DIR)/data_from_isolated_speaker.csv" VAL_SPLIT="$(VAL_SPLIT)" BUFFER_SECONDS="$(BUFFER_SECONDS)" WHISPER_MODEL="$(WHISPER_MODEL)" COMPUTE_TYPE="$(COMPUTE_TYPE)" MAX_SEGMENT_SECONDS_DATASET="$(MAX_SEGMENT_SECONDS_DATASET)" MAX_TEXT_CHARS="$(MAX_TEXT_CHARS)" PRE_ASR_MAX_CHUNK_SECONDS="$(PRE_ASR_MAX_CHUNK_SECONDS)" PRE_ASR_MIN_CHUNK_SECONDS="$(PRE_ASR_MIN_CHUNK_SECONDS)" PRE_ASR_MIN_SILENCE_MS="$(PRE_ASR_MIN_SILENCE_MS)" PRE_ASR_KEEP_SILENCE_MS="$(PRE_ASR_KEEP_SILENCE_MS)" PRE_ASR_MERGE_GAP_MS="$(PRE_ASR_MERGE_GAP_MS)" PRE_ASR_SILENCE_THRESH_DB="$(PRE_ASR_SILENCE_THRESH_DB)"
	$(MAKE) review-dataset VOICE="$(VOICE)" WORKSPACE="$(WORKSPACE)" DATASET_DIR="$(DATASET_DIR)" TRAIN_CSV="$(DATASET_DIR)/metadata_train.csv" REVIEW_CSV="$(REVIEW_CSV)" CURATED_CSV="$(CURATED_CSV)" MIN_AUDIO_SECONDS="$(MIN_AUDIO_SECONDS)" MAX_AUDIO_SECONDS_REVIEW="$(MAX_AUDIO_SECONDS_REVIEW)" MIN_TEXT_CHARS="$(MIN_TEXT_CHARS)" MAX_TEXT_CHARS_REVIEW="$(MAX_TEXT_CHARS_REVIEW)" MIN_CHARS_PER_SECOND="$(MIN_CHARS_PER_SECOND)" MAX_CHARS_PER_SECOND="$(MAX_CHARS_PER_SECOND)"

train:
	@if [ ! -f "$(TRAIN_CSV)" ] || [ ! -f "$(EVAL_CSV)" ]; then echo "Use: make train TRAIN_CSV=... EVAL_CSV=..."; exit 1; fi
	$(PY) cli.py train --train-csv $(TRAIN_CSV) --eval-csv $(EVAL_CSV) --language $(TTS_LANGUAGE) --epochs $(or $(EPOCHS),10) --batch-size $(or $(BATCH_SIZE),4) --grad-accumm $(or $(GRAD_ACCUMM),1) --max-audio-seconds $(or $(MAX_AUDIO_SECONDS_TRAIN),11) --max-text-chars $(or $(MAX_TEXT_CHARS),200) --output-dir $(or $(TRAIN_OUTPUT_DIR),$(WORKSPACE)/training)

train-curated:
	@if [ ! -f "$(CURATED_CSV)" ] || [ ! -f "$(EVAL_CSV)" ]; then echo "Use: make train-curated CURATED_CSV=... EVAL_CSV=..."; exit 1; fi
	$(PY) cli.py train --train-csv $(CURATED_CSV) --eval-csv $(EVAL_CSV) --language $(TTS_LANGUAGE) --epochs $(or $(EPOCHS),10) --batch-size $(or $(BATCH_SIZE),4) --grad-accumm $(or $(GRAD_ACCUMM),1) --max-audio-seconds $(or $(MAX_AUDIO_SECONDS_TRAIN),11) --max-text-chars $(or $(MAX_TEXT_CHARS),200) --output-dir $(or $(TRAIN_OUTPUT_DIR),$(WORKSPACE)/training)

infer:
	@if [ -z "$(TEXT)" ] || [ -z "$(VOICE_MODEL)" ]; then echo "Use: make infer VOICE_MODEL=silvio TEXT='olá' OUTPUT=out.wav"; exit 1; fi
	$(PY) cli.py infer --voice-model $(VOICE_MODEL) --text "$(TEXT)" --output $(or $(OUTPUT),output.wav) --voices-config $(or $(VOICES_CONFIG),config/voices.json) $(if $(TEMPERATURE),--temperature $(TEMPERATURE),) $(if $(SPEED),--speed $(SPEED),) $(if $(TTS_LANGUAGE),--language $(TTS_LANGUAGE),)

download:
	@if [ -z "$(URL)" ] || [ -z "$(FILENAME)" ]; then echo "Use: make download URL=... FILENAME=..."; exit 1; fi
	$(PY) cli.py download --url "$(URL)" --filename "$(FILENAME)" --output-folder $(or $(OUTPUT_DIR),.) --trim-start $(or $(TRIM_START),0) --trim-end $(or $(TRIM_END),0)

enhance:
	@if [ -z "$(INPUT)" ]; then echo "Use: make enhance INPUT=input.wav OUTPUT=enhanced.wav"; exit 1; fi
	$(PY) cli.py enhance --input "$(INPUT)" --output $(or $(OUTPUT),enhanced.wav)

serve:
	$(PY) server.py --host $(or $(HOST),0.0.0.0) --port $(or $(PORT),8080) --voices-config $(or $(VOICES_CONFIG),config/voices.json)

clean:
	find . -type d -name '__pycache__' -prune -exec rm -r {} +
	[ -d outputs ] && rm -r outputs || true
