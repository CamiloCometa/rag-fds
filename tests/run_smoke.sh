#!/usr/bin/env bash
# Prueba de integración del código con FDS sintéticas y un Ollama FALSO (no necesita modelos).
# Verifica que extracción → verificación → índice → RAG → RAGAS → reporte corren sin errores.
set -euo pipefail
cd "$(dirname "$0")/.."
export RAG_FDS_CONFIG=tests/config_test.yaml
rm -rf tests/_out
python tests/make_synthetic_fds.py tests/fixtures/pdfs
python tests/mock_ollama.py 11434 & MOCK=$!
trap 'kill $MOCK' EXIT
sleep 1
python -m src.extract
python -m src.verify
python -m src.index
python -m src.rag "¿Cuál es el punto de inflamación del Esmalte Sintético Blanco?" --k 3
python -m src.evaluate all
echo "OK: pipeline completo ejecutado (los puntajes del mock no tienen significado)."
