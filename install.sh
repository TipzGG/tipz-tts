#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "error: ffmpeg/ffprobe nao encontrados no PATH. Instale FFmpeg antes de continuar." >&2
  exit 1
fi

"$ROOT_DIR/scripts/setup_venv.sh"

echo
echo "Instalacao concluida."
echo "Use:"
echo "  source .venv/bin/activate"
