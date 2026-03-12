PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: venv install test pipeline train infer download enhance serve clean

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements-dev.txt

test:
	$(PY) -m unittest discover -s tests -p 'test_*.py'

pipeline:
	@if [ -z "$(CONFIG)" ]; then echo "Use: make pipeline CONFIG=templates/minha_voz.json"; exit 1; fi
	$(PY) cli.py pipeline --config $(CONFIG)

train:
	@if [ -z "$(TRAIN_CSV)" ] || [ -z "$(EVAL_CSV)" ]; then echo "Use: make train TRAIN_CSV=... EVAL_CSV=..."; exit 1; fi
	$(PY) cli.py train --train-csv $(TRAIN_CSV) --eval-csv $(EVAL_CSV)

infer:
	@if [ -z "$(TEXT)" ] || [ -z "$(VOICE_MODEL)" ]; then echo "Use: make infer VOICE_MODEL=silvio TEXT='olá' OUTPUT=out.wav"; exit 1; fi
	$(PY) cli.py infer --voice-model $(VOICE_MODEL) --text "$(TEXT)" --output $(or $(OUTPUT),output.wav) --voices-config $(or $(VOICES_CONFIG),config/voices.json)

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
