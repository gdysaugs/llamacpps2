#!/bin/bash
set -e

echo "🎭🎬🤖 Starting Wav2Lip+SoVITS+Llama integrated app on Docker..."
echo "Working directory: $(pwd)"
echo "PORT: ${PORT:-8080}"

# Python Path設定
export PYTHONPATH="/app:$PYTHONPATH"

# 統合アプリケーション起動
cd /app/gradio_frontend

# 各venvのsite-packagesをPYTHONPATHに追加
export PYTHONPATH="/app/gradio_venv/lib/python3.10/site-packages:/app/venv/lib/python3.10/site-packages:/app/sovits_venv/lib/python3.10/site-packages:/app/facefusion_venv/lib/python3.10/site-packages:/app/llama_venv/lib/python3.10/site-packages:$PYTHONPATH"

# メインはgradio_venvでアクティベート
source /app/gradio_venv/bin/activate

# GPU確認
nvidia-smi || echo "GPU not available"

echo "Starting application with PYTHONPATH: $PYTHONPATH"

python wav2lip_sovits_llama_integrated.py \
    --server-name 0.0.0.0 \
    --server-port ${PORT:-8080} \
    --share=False