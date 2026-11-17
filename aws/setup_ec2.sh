#!/usr/bin/env bash
# Prepara una instancia EC2 (Ubuntu 22.04/24.04, x86_64) para correr TODO el proyecto:
# Tesseract (spa), Python 3.12 (vía uv), dependencias fijadas, Ollama y los modelos.
#
# Uso (en la instancia, dentro de la carpeta del repo):
#     bash aws/setup_ec2.sh                 # modelos por defecto de config.yaml
#     GEN_MODEL=qwen2.5:3b-instruct bash aws/setup_ec2.sh   # instancia sin GPU y poca RAM
set -euo pipefail
cd "$(dirname "$0")/.."

GEN_MODEL="${GEN_MODEL:-qwen2.5:7b-instruct}"
EMB_MODEL="${EMB_MODEL:-bge-m3}"

echo "== Paquetes del sistema (Tesseract español, utilidades)"
sudo apt-get update -y
sudo apt-get install -y tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng tmux unzip curl

echo "== Python 3.12 + entorno virtual (uv)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-demo.txt
.venv/bin/python -m pip freeze > requirements-lock.txt 2>/dev/null || uv pip freeze --python .venv/bin/python > requirements-lock.txt

echo "== Ollama"
if ! command -v ollama >/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
# Contexto de 8192 tokens: por la API compatible OpenAI (la que usa RAGAS) no se puede fijar num_ctx
# por petición; sin esto Ollama recorta en silencio los prompts largos del evaluador.
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_CONTEXT_LENGTH=8192"\nEnvironment="OLLAMA_NUM_PARALLEL=1"\nEnvironment="OLLAMA_KEEP_ALIVE=24h"\n' \
  | sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl restart ollama
sleep 3
ollama pull "$EMB_MODEL"
ollama pull "$GEN_MODEL"

if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  echo "GPU detectada:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "AVISO: sin GPU. Funciona en CPU pero más lento; con <16 GB RAM use GEN_MODEL=qwen2.5:3b-instruct"
fi

if [ "$GEN_MODEL" != "qwen2.5:7b-instruct" ]; then
  sed -i "s#model: qwen2.5:7b-instruct #model: $GEN_MODEL #; s#evaluator_model: qwen2.5:7b-instruct#evaluator_model: $GEN_MODEL#" config.yaml
  echo "config.yaml actualizado para usar $GEN_MODEL (generador y evaluador)"
fi

echo
echo "Listo. Active el entorno con:  source .venv/bin/activate"
echo "Corra los pasos largos dentro de tmux (sobreviven a la desconexión SSH):  tmux new -s rag"
