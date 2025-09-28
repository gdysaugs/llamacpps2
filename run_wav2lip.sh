#!/bin/bash
# SoVITS-Wav2Lip-LlamaCPP統合システム ポータブル版起動スクリプト
# Linux/macOS用シェルスクリプト

clear
echo "======================================================"
echo "  SoVITS-Wav2Lip-LlamaCPP統合システム ポータブル版"
echo "======================================================"
echo

# 現在のディレクトリをアプリケーションルートに設定
cd "$(dirname "$0")"

# 環境変数設定
export PYTHONPATH="$PWD:$PWD/gradio_frontend:$PWD/python/lib/python3.10/site-packages"
export PATH="$PWD/python/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/python/lib:$LD_LIBRARY_PATH"

# Python実行ファイルのパスを設定
if [ -f "python/bin/python" ]; then
    PYTHON_EXE="python/bin/python"
elif [ -f "python/python" ]; then
    PYTHON_EXE="python/python"
else
    PYTHON_EXE="python3"
fi

echo "🐍 Python実行ファイル: $PYTHON_EXE"

# CUDA環境確認
echo "🔍 CUDA環境確認中..."
$PYTHON_EXE -c "import torch; print('CUDA利用可能:', torch.cuda.is_available()); print('GPU数:', torch.cuda.device_count())" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  PyTorchまたはCUDAが利用できません。CPUモードで動作します。"
else
    echo "✅ CUDA環境確認完了"
fi
echo

# 必要なディレクトリ作成
mkdir -p output
mkdir -p temp
mkdir -p models

echo "📁 ディレクトリ構造確認完了"
echo

# モデルファイル確認
echo "🔍 モデルファイル確認中..."
if [ -f "models/wav2lip_gan.pth" ]; then
    echo "✅ Wav2Lipモデル: 確認済み"
else
    echo "❌ Wav2Lipモデル: models/wav2lip_gan.pth が見つかりません"
    echo "   初回起動時は自動ダウンロードが実行されます"
fi

if [ -d "models/gpt_sovits" ]; then
    echo "✅ GPT-SoVITSモデル: 確認済み"
else
    echo "❌ GPT-SoVITSモデル: models/gpt_sovits が見つかりません"
fi
echo

# 実行権限の確認・設定
chmod +x "$0" 2>/dev/null

# Pythonスクリプト実行
echo "🚀 アプリケーション起動中..."
echo "   Web UI: http://localhost:7866"
echo "   終了するには Ctrl+C を押してください"
echo

$PYTHON_EXE gradio_frontend/wav2lip_sovits_llama_integrated_portable.py

echo
echo "アプリケーションが終了しました。"
read -p "何かキーを押してください..."