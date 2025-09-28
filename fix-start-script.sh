#!/bin/bash

# 修正されたスタートアップスクリプトを作成
cat > start-fixed.sh <<'EOF'
#!/bin/bash
set -e

echo "Starting integrated Wav2lip+SoVITS+Llama on Cloud Run GPU..."

# 統合アプリケーション起動
cd /app/gradio_frontend && \
source /app/gradio_venv/bin/activate && \
python wav2lip_sovits_llama_integrated.py \
    --server-name 0.0.0.0 \
    --server-port ${PORT:-8080} \
    --share=False
EOF

chmod +x start-fixed.sh

echo "Fixed start script created: start-fixed.sh"