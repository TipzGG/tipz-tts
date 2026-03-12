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
./scripts/setup_venv.sh
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
python cli.py train --train-csv output/metadata_train.csv --eval-csv output/metadata_eval.csv
python cli.py infer --voice-model silvio --text "Olá" --output output.wav --voices-config config/voices.json
```

### Isolar speaker (quando ha mais de uma pessoa falando)

Instale antes:

```bash
python -m pip install pyannote.audio
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
