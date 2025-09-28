#!/bin/bash

echo "========================================="
echo "wav2lip-project 復元スクリプト"
echo "========================================="
echo ""

# バックアップの存在確認
BACKUP_DIR="/mnt/c/wav2lip_backup"
if [ ! -d "$BACKUP_DIR" ]; then
    echo "エラー: バックアップディレクトリが見つかりません: $BACKUP_DIR"
    echo "SETUP_AFTER_CLONE.md を参照してバックアップを作成してください。"
    exit 1
fi

echo "✓ バックアップディレクトリ確認: $BACKUP_DIR"
echo ""

# 1. モデルファイルの復元
echo "========================================="
echo "1. モデルファイルを復元中..."
echo "========================================="

if [ -d "$BACKUP_DIR/models" ]; then
    echo "models/ フォルダをコピー中（これには時間がかかります）..."
    cp -r "$BACKUP_DIR/models" ./
    echo "✓ models/ フォルダを復元しました"
else
    echo "⚠ 警告: $BACKUP_DIR/models が見つかりません"
fi

echo ""

# 2. 大きなモデルファイルの復元
echo "========================================="
echo "2. 大きなモデルファイルを復元中..."
echo "========================================="

mkdir -p GPT_SoVITS/pretrained_models/sv utils faceID

if [ -f "$BACKUP_DIR/large_models/pretrained_eres2netv2w24s4ep4.ckpt" ]; then
    cp "$BACKUP_DIR/large_models/pretrained_eres2netv2w24s4ep4.ckpt" GPT_SoVITS/pretrained_models/sv/
    echo "✓ pretrained_eres2netv2w24s4ep4.ckpt を復元しました"
else
    echo "⚠ 警告: pretrained_eres2netv2w24s4ep4.ckpt が見つかりません"
fi

if [ -f "$BACKUP_DIR/large_models/detection_Resnet50_Final.pth" ]; then
    cp "$BACKUP_DIR/large_models/detection_Resnet50_Final.pth" utils/
    echo "✓ detection_Resnet50_Final.pth を復元しました"
else
    echo "⚠ 警告: detection_Resnet50_Final.pth が見つかりません"
fi

if [ -f "$BACKUP_DIR/large_models/recognition.onnx" ]; then
    cp "$BACKUP_DIR/large_models/recognition.onnx" faceID/
    echo "✓ recognition.onnx を復元しました"
else
    echo "⚠ 警告: recognition.onnx が見つかりません"
fi

echo ""

# 3. 仮想環境のセットアップ
echo "========================================="
echo "3. 仮想環境を作成中..."
echo "========================================="
echo "これには数分から数十分かかります。"
echo ""

# メインのvenv
if [ ! -d "venv" ]; then
    echo "venv を作成中..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    deactivate
    echo "✓ venv を作成しました"
else
    echo "✓ venv は既に存在します"
fi

echo ""

# facefusion_env
if [ ! -d "facefusion_env" ]; then
    echo "facefusion_env を作成中..."
    python3 -m venv facefusion_env
    source facefusion_env/bin/activate
    pip install --upgrade pip
    pip install -r requirements_facefusion.txt
    deactivate
    echo "✓ facefusion_env を作成しました"
else
    echo "✓ facefusion_env は既に存在します"
fi

echo ""

# gpt_sovits_env
if [ ! -d "gpt_sovits_env" ]; then
    echo "gpt_sovits_env を作成中..."
    python3 -m venv gpt_sovits_env
    source gpt_sovits_env/bin/activate
    pip install --upgrade pip
    pip install -r requirements_gpt_sovits.txt
    deactivate
    echo "✓ gpt_sovits_env を作成しました"
else
    echo "✓ gpt_sovits_env は既に存在します"
fi

echo ""

# llama_venv
if [ ! -d "llama_venv" ]; then
    echo "llama_venv を作成中..."
    python3 -m venv llama_venv
    source llama_venv/bin/activate
    pip install --upgrade pip
    pip install llama-cpp-python
    deactivate
    echo "✓ llama_venv を作成しました"
else
    echo "✓ llama_venv は既に存在します"
fi

echo ""

# gradio_frontend/gradio_venv
if [ ! -d "gradio_frontend/gradio_venv" ]; then
    echo "gradio_frontend/gradio_venv を作成中..."
    cd gradio_frontend
    python3 -m venv gradio_venv
    source gradio_venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    deactivate
    cd ..
    echo "✓ gradio_frontend/gradio_venv を作成しました"
else
    echo "✓ gradio_frontend/gradio_venv は既に存在します"
fi

echo ""
echo "========================================="
echo "✓ 復元完了！"
echo "========================================="
echo ""
echo "動作確認："
echo "  source venv/bin/activate"
echo "  python wav2lip_inference.py --help"
echo ""
echo "または gradio インターフェースを起動："
echo "  cd gradio_frontend"
echo "  source gradio_venv/bin/activate"
echo "  python wav2lip_sovits_llama_integrated.py"
echo ""