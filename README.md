# tipz-tts

Projeto com treino/pipeline offline via `cli.py` e API HTTP online apenas para síntese via `server.py`.

## Estrutura

- `src/app/`: lógica de negócio (`audio`, `dataset`, `training`, `inference`, `pipeline`, `profiles`).
- `cli.py`: comandos de linha (`download`, `enhance`, `dataset`, `train`, `infer`, `pipeline`).
- `cli.py`: comandos de linha (`download`, `enhance`, `import-inputs`, `isolate-speaker`, `dataset`, `train`, `infer`, `pipeline`).
- `server.py`: API HTTP Flask para TTS.
- `config/voices.example.json`: exemplo de registro de vozes.
- `templates/voice_template.json`: template da pipeline por voz.
- `templates/data_example.csv`: exemplo de CSV para dataset/treino.
- `tests/`: testes unitários.

## Setup

```bash
python3.11 --version
./scripts/setup_venv.sh
source .venv/bin/activate
```

Observacao: este projeto requer Python `3.11` por causa das dependencias `trainer`/`coqui-tts`.
O ambiente base nao instala `pyannote.audio`; ele ficou opcional porque o resolver do `pip` pode estourar profundidade ao tentar fechar toda a arvore junto com o resto do stack.
O setup remove automaticamente o pacote legado `coqpit` e reinstala `coqpit-config`, para evitar conflito com `coqui-tts`.

Atalho para outra maquina:

```bash
./install.sh
source .venv/bin/activate
```

Dependencias de sistema (obrigatorio para audio/video local com `pydub`):

```bash
brew install ffmpeg
```

Ou:

```bash
make install
```

## Registro de vozes (sem env por voz)

1. Copie:

```bash
cp config/voices.example.json config/voices.json
```

2. Preencha os caminhos reais dos modelos/checkpoints/referências.
3. Use `status: "enabled"` para vozes expostas na API online.
4. Use `status: "disabled"` para vozes não publicadas.

5. Quando o servidor subir, ele carrega apenas vozes `enabled` no runtime de síntese.

## CLI

```bash
python cli.py pipeline --config templates/voice_template.json
python cli.py import-inputs --input-dir inputs --output-folder outputs/local --speaker silvio --skip-enhancement
python cli.py isolate-speaker --input-dir inputs --output-dir outputs/isolated --speaker silvio --reference-audio inputs/ref.wav
python cli.py review-dataset --metadata-csv output/metadata_train.csv --filtered-output-csv output/metadata_train_curated.csv
python cli.py train --train-csv output/metadata_train.csv --eval-csv output/metadata_eval.csv
python cli.py infer --voice-model silvio --text "Olá" --output output.wav --voices-config config/voices.json
```

## Pipelines via Makefile

O `Makefile` agora expõe dois fluxos agnosticos por voz:

- `pipeline-local`: para audio com um speaker principal, sem diarizacao
- `pipeline-isolated`: para audio com multiplas vozes, usando diarizacao + selecao do speaker alvo
- `pipeline-youtube-auto`: para multiplos videos do YouTube via arquivo de config, com dataset automatico e sem revisao manual

Variaveis principais:

- `VOICE`: nome logico da voz
- `INPUT_DIR`: pasta com os audios brutos
- `WORKSPACE`: pasta de saida da voz
- `TTS_LANGUAGE`: idioma do speaker

Inspecionar defaults:

```bash
make print-pipeline-vars VOICE=beerschool
```

### Fluxo 1: audio direto

Importa os arquivos locais, melhora o audio, gera dataset e cria uma revisao automatica para curadoria.

```bash
make pipeline-local VOICE=beerschool INPUT_DIR=inputs/beerschool TTS_LANGUAGE=pt
```

Saidas principais:

- `outputs/beerschool/raw/data_from_inputs.csv`
- `outputs/beerschool/dataset/metadata_train.csv`
- `outputs/beerschool/dataset/metadata_eval.csv`
- `outputs/beerschool/dataset/metadata_train_review.csv`
- `outputs/beerschool/dataset/metadata_train_curated.csv`

### Fluxo 2: isolamento de speaker

Use quando o audio tiver mais de uma pessoa falando.

Instale antes:

```bash
make install-speaker-isolation
```

Exemplo com audio de referencia:

```bash
make pipeline-isolated \
  VOICE=beerschool \
  INPUT_DIR=inputs/beerschool \
  TTS_LANGUAGE=pt \
  REFERENCE_AUDIO=inputs/beerschool_ref.wav \
  HF_TOKEN=seu_token
```

Saidas principais:

- `outputs/beerschool/isolated/data_from_isolated_speaker.csv`
- `outputs/beerschool/dataset/metadata_train.csv`
- `outputs/beerschool/dataset/metadata_eval.csv`
- `outputs/beerschool/dataset/metadata_train_review.csv`
- `outputs/beerschool/dataset/metadata_train_curated.csv`

### Curadoria e treino

Depois de revisar o `metadata_train_review.csv`, treine com o dataset curado:

```bash
make train-curated VOICE=beerschool TTS_LANGUAGE=pt
```

Se quiser treinar sem a curadoria filtrada:

```bash
make train VOICE=beerschool TTS_LANGUAGE=pt
```

Observacoes de treino:
- O treino agora filtra automaticamente linhas com texto acima de `--max-text-chars` (default `180`) e grava CSVs preparados em `.../run/prepared/`.
- O CSV preparado converte `audio_file` relativo para path absoluto automaticamente (nao precisa symlink manual de `wavs/`).
- Se usar CLI direto, `--grad-accum` e `--grad-accumm` funcionam como alias.
- Para continuar um treino anterior, use `--restore-path` (checkpoint especifico) ou `--resume-latest` (ultimo checkpoint encontrado no output).

Exemplo recomendado:

```bash
python cli.py train \
  --train-csv outputs/beerschool/dataset/metadata_train_auto.csv \
  --eval-csv outputs/beerschool/dataset/metadata_eval_auto.csv \
  --language pt \
  --epochs 15 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-audio-seconds 11 \
  --max-text-chars 180 \
  --output-dir outputs/beerschool/training_test
```

Exemplos de resume:

```bash
python cli.py train \
  --train-csv outputs/beerschool/dataset/metadata_train_auto.csv \
  --eval-csv outputs/beerschool/dataset/metadata_eval_auto.csv \
  --output-dir outputs/beerschool/training_test \
  --resume-latest
```

```bash
python cli.py train \
  --train-csv outputs/beerschool/dataset/metadata_train_auto.csv \
  --eval-csv outputs/beerschool/dataset/metadata_eval_auto.csv \
  --output-dir outputs/beerschool/training_test \
  --restore-path outputs/beerschool/training_test/run/training/GPT_XTTS_FT-<timestamp>/checkpoint_150.pth
```

Treino estilo "friend script" (defaults proximos do script classico do Coqui):

```bash
make train-friend \
  VOICE=beerschool \
  TRAIN_CSV=outputs/beerschool/dataset/metadata_train_curated.csv \
  EVAL_CSV=outputs/beerschool/dataset/metadata_eval.csv \
  TRAIN_OUTPUT_DIR=outputs/beerschool/training_friend
```

### Reports e checkpoint

Resumo de qualidade do dataset:

```bash
./report.sh outputs/beerschool/dataset
```

Ranking de checkpoints (tenta usar loss do TensorBoard; fallback para ultimo checkpoint):

```bash
make checkpoint-report WORKSPACE=outputs/beerschool
```

Ou por run especifico:

```bash
python scripts/checkpoint_report.py \
  --run-dir outputs/beerschool/training_test/run/training/GPT_XTTS_FT-<timestamp>
```

### YouTube-first automatico

Para RunPod/Vast, o fluxo mais indicado e usar um arquivo `CONFIG` com varias URLs de YouTube e deixar o pipeline exportar o dataset automatico.

Exemplo:

```bash
make pipeline-youtube-auto CONFIG=templates/beerschool.json
```

Esse fluxo:
- baixa varias fontes de YouTube
- converte para WAV
- faz chunking pre-ASR
- gera `metadata_train.csv` e `metadata_eval.csv`
- pontua cada chunk automaticamente
- exporta `metadata_train_auto.csv` e `metadata_eval_auto.csv`

Arquivos principais:
- `metadata_train_scored.csv`
- `metadata_eval_scored.csv`
- `metadata_train_auto.csv`
- `metadata_eval_auto.csv`

Configuracao relevante no `CONFIG`:
- `sources`: lista de videos do YouTube
- `dataset.whisper_model`
- `dataset.compute_type`
- `dataset.pre_asr_*`
- `dataset.auto_curate.policy`
- `train.use_auto_dataset`

### Curadoria do dataset

Depois de gerar `metadata_train.csv`, rode uma revisao automatica para localizar trechos suspeitos:

```bash
python cli.py review-dataset \
  --metadata-csv outputs/dataset/beerschool/metadata_train.csv \
  --filtered-output-csv outputs/dataset/beerschool/metadata_train_curated.csv
```

Saidas:
- `*_review.csv`: adiciona `duration_seconds`, `chars_per_second`, `flags` e `review_status`
- `*_curated.csv`: exporta apenas linhas marcadas automaticamente como `keep`

Flags uteis para revisao manual:
- `audio_too_short` / `audio_too_long`
- `text_too_short` / `text_too_long`
- `chars_per_second_low` / `chars_per_second_high`
- `ellipsis`, `many_commas`, `filler_words`, `suspicious_chars`
- `no_terminal_punctuation`

Fluxo recomendado:
- abra primeiro o `*_review.csv`
- corrija ou remova linhas marcadas como `review`
- use o `*_curated.csv` como base inicial para um treino mais limpo

### Chunking pre-ASR

O passo `dataset` agora quebra arquivos longos em chunks de fala antes de rodar o Whisper. Isso reduz uso de memoria e melhora a estabilidade em maquinas locais e cloud.

Parametros mais uteis:

- `--pre-asr-max-chunk-seconds`: tamanho maximo de cada bloco antes do ASR
- `--pre-asr-min-chunk-seconds`: descarta blocos pequenos demais
- `--pre-asr-min-silence-ms`: duracao minima de silencio para separar fala
- `--pre-asr-keep-silence-ms`: margem mantida nas bordas de cada bloco
- `--pre-asr-merge-gap-ms`: junta blocos de fala muito proximos
- `--pre-asr-silence-thresh-db`: limiar de silencio do detector

Exemplo:

```bash
python cli.py dataset \
  --input-csv outputs/beerschool/raw/data_from_inputs.csv \
  --output-dir outputs/beerschool/dataset \
  --whisper-model medium \
  --pre-asr-max-chunk-seconds 30 \
  --pre-asr-min-silence-ms 600
```

### Isolar speaker (quando ha mais de uma pessoa falando)

Instale antes:

```bash
python -m pip install -r requirements-speaker-isolation.txt
```

Exemplo:

```bash
python cli.py isolate-speaker \
  --input-dir inputs \
  --output-dir outputs/isolated/beerschool \
  --speaker beerschool \
  --reference-audio inputs/beerschool_ref.wav
```

Saidas:
- `outputs/isolated/beerschool/wavs/*.wav` (trechos somente do speaker alvo)
- `outputs/isolated/beerschool/data_from_isolated_speaker.csv` (pronto para `cli.py dataset`)

## Server HTTP (inferência online)

```bash
python server.py --host 0.0.0.0 --port 8080 --voices-config config/voices.json
```

### Health

`GET /health`
`GET /healthz`

### List voices

`GET /voices`
`GET /v1/voices`
`GET /v1/voices/{voice_id}`

### TTS (legado)

`POST /` ou `POST /tts` com JSON:

```json
{
  "text": "olá mundo",
  "voice_model": "silvio",
  "language": "pt",
  "speaking_rate": 0.97
}
```

Resposta: `audio/wav` (binário).

### TTS v1 (canônico)

`POST /v1/tts/synthesize`

```json
{
  "voice_id": "silvio",
  "text": "olá mundo",
  "language": "pt-BR",
  "format": "wav",
  "sample_rate": 24000,
  "speaking_rate": 0.97,
  "temperature": 0.8,
  "request_id": "req-123"
}
```

Headers de resposta:

- `X-Voice-Id`
- `X-Request-Id`

Erros padronizados:

- `INVALID_ARGUMENT` (400)
- `VOICE_NOT_FOUND` (404)
- `TEXT_TOO_LARGE` (413)
- `UNSUPPORTED_FORMAT` (415)
- `SYNTHESIS_FAILED` (422)
- `RATE_LIMITED` (429)
- `INTERNAL_ERROR` (500)

## Operação (server)

Variáveis opcionais:

- `TTS_INFERENCE_TIMEOUT_SECONDS` (default: `45`)
- `TTS_MAX_WORKERS` (default: `2`)
- `TTS_RATE_LIMIT_RPM` (default: `0`, desabilitado)
- `LOG_LEVEL` (default: `INFO`)

Para download do YouTube no pipeline offline (anti-bot):

- `YT_DLP_COOKIES_FROM_BROWSER` (ex.: `chrome`, `firefox`)
- `YT_DLP_COOKIES_FILE` (path de cookies exportado)

## Testes

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```
