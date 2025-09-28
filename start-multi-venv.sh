#!/bin/bash
set -e

echo "Starting integrated Wav2lip+SoVITS+Llama with multi-venv support..."

# 複数venvのPYTHONPATHを設定
export PYTHONPATH="/app/venv/lib/python3.10/site-packages:/app/gradio_venv/lib/python3.10/site-packages:/app/sovits_venv/lib/python3.10/site-packages:/app/facefusion_venv/lib/python3.10/site-packages:/app/llama_venv/lib/python3.10/site-packages:/app:$PYTHONPATH"

# メインvenvをアクティベート（PyTorchとML系パッケージ）
source /app/venv/bin/activate

# Gradioを追加インストール
echo "Installing gradio in main venv..."
pip install --no-cache-dir gradio>=5.0.0

cd /app/gradio_frontend && \
python wav2lip_sovits_llama_integrated.py \
    --server-name 0.0.0.0 \
    --server-port ${PORT:-8080} \
    --share=False