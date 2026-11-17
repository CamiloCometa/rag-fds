#!/usr/bin/env bash
# Modelos para CPU (AWS Learner Lab, t3.large sin GPU):
#  - granite-embedding:278m (embeddings multilingües, más liviano que bge-m3)
#  - qwen2.5-3b-rag: qwen2.5:3b-instruct con 2 hilos y contexto de 8192 tokens.
#    Necesario porque RAGAS llama a Ollama por la API compatible OpenAI, donde no se puede
#    fijar num_thread ni num_ctx por petición; quedan fijados en el propio modelo.
set -euo pipefail
ollama pull granite-embedding:278m
ollama pull qwen2.5:3b-instruct
cat > /tmp/Modelfile.qwen <<'MF'
FROM qwen2.5:3b-instruct
PARAMETER num_thread 2
PARAMETER num_ctx 8192
PARAMETER temperature 0
MF
ollama create qwen2.5-3b-rag -f /tmp/Modelfile.qwen
ollama list
