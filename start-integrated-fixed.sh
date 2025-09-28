#!/bin/bash
set -e

echo "🎭🎬🤖 Starting Wav2Lip+SoVITS+Llama integrated app (fixed version)..."
echo "Working directory: $(pwd)"

# Python Path設定
export PYTHONPATH="/home/adama/wav2lip-project:$PYTHONPATH"

# 統合アプリケーション起動
cd /home/adama/wav2lip-project/gradio_frontend

echo "Starting application with PORT=${PORT:-7866}..."

python3 wav2lip_sovits_llama_integrated.py \
    --server-name 0.0.0.0 \
    --server-port ${PORT:-7866} \
    --share=True \
    --debug