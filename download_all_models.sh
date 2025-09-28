#!/bin/bash

echo "========================================="
echo "全モデルファイル ダウンロードスクリプト"
echo "========================================="
echo ""

# ディレクトリ作成
mkdir -p models/wav2lip models/gpt_sovits models/facefusion models/retinaface models/gfpgan
mkdir -p GPT_SoVITS/pretrained_models/sv utils faceID

echo "1. Wav2Lip モデルをダウンロード中..."
echo "----------------------------------------"

# Wav2Lip GAN model
if [ ! -f "models/wav2lip/wav2lip_gan.pth" ]; then
    echo "wav2lip_gan.pth をダウンロード中..."
    wget -O models/wav2lip/wav2lip_gan.pth https://github.com/Rudrabha/Wav2Lip/releases/download/models/wav2lip_gan.pth
    echo "✓ wav2lip_gan.pth"
else
    echo "✓ wav2lip_gan.pth は既に存在します"
fi

# Wav2Lip model (no GAN)
if [ ! -f "models/wav2lip/wav2lip.pth" ]; then
    echo "wav2lip.pth をダウンロード中..."
    wget -O models/wav2lip/wav2lip.pth https://github.com/Rudrabha/Wav2Lip/releases/download/models/wav2lip.pth
    echo "✓ wav2lip.pth"
else
    echo "✓ wav2lip.pth は既に存在します"
fi

echo ""
echo "2. Face Detection モデルをダウンロード中..."
echo "----------------------------------------"

# s3fd face detector
if [ ! -f "models/detection_Resnet50_Final.pth" ]; then
    echo "detection_Resnet50_Final.pth をダウンロード中 (104MB)..."
    wget -O models/detection_Resnet50_Final.pth https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
    cp models/detection_Resnet50_Final.pth utils/detection_Resnet50_Final.pth
    echo "✓ detection_Resnet50_Final.pth"
else
    echo "✓ detection_Resnet50_Final.pth は既に存在します"
fi

echo ""
echo "3. RetinaFace モデルをダウンロード中..."
echo "----------------------------------------"

# RetinaFace model
if [ ! -f "models/retinaface/Resnet50_Final.pth" ]; then
    echo "RetinaFace Resnet50_Final.pth をダウンロード中..."
    wget -O models/retinaface/Resnet50_Final.pth https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
    echo "✓ RetinaFace Resnet50_Final.pth"
else
    echo "✓ RetinaFace Resnet50_Final.pth は既に存在します"
fi

echo ""
echo "4. FaceFusion / InsightFace モデルをダウンロード中..."
echo "----------------------------------------"

# inswapper model
if [ ! -f "models/facefusion/inswapper_128.onnx" ]; then
    echo "inswapper_128.onnx をダウンロード中 (530MB)..."
    wget -O models/facefusion/inswapper_128.onnx https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx
    echo "✓ inswapper_128.onnx"
else
    echo "✓ inswapper_128.onnx は既に存在します"
fi

# face recognition model
if [ ! -f "faceID/recognition.onnx" ]; then
    echo "recognition.onnx をダウンロード中 (91MB)..."
    wget -O faceID/recognition.onnx https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_w600k_r50.onnx
    echo "✓ recognition.onnx"
else
    echo "✓ recognition.onnx は既に存在します"
fi

echo ""
echo "5. GFPGAN (顔品質向上) モデルをダウンロード中..."
echo "----------------------------------------"

# GFPGAN model
if [ ! -f "models/gfpgan/GFPGANv1.4.pth" ]; then
    echo "GFPGANv1.4.pth をダウンロード中..."
    wget -O models/gfpgan/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
    echo "✓ GFPGANv1.4.pth"
else
    echo "✓ GFPGANv1.4.pth は既に存在します"
fi

echo ""
echo "6. GPT-SoVITS モデルをダウンロード中..."
echo "----------------------------------------"

# GPT-SoVITS pretrained models
if [ ! -f "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt" ]; then
    echo "pretrained_eres2netv2w24s4ep4.ckpt をダウンロード中 (102MB)..."
    wget -O GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt \
        https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth
    echo "✓ pretrained_eres2netv2w24s4ep4.ckpt"
else
    echo "✓ pretrained_eres2netv2w24s4ep4.ckpt は既に存在します"
fi

# Use Python script for HuggingFace models
if command -v python3 &> /dev/null; then
    echo ""
    echo "7. Python経由でHuggingFaceモデルをダウンロード中..."
    echo "----------------------------------------"
    python3 download_models.py
fi

echo ""
echo "========================================="
echo "✓ ダウンロード完了！"
echo "========================================="
echo ""
echo "ダウンロードされたモデル："
du -sh models/* 2>/dev/null | grep -v "cannot access"
echo ""
echo "合計サイズ："
du -sh models 2>/dev/null
echo ""
echo "注意："
echo "  - Llama 3.7GBモデルは除外されています"
echo "  - 必要な場合は別途ダウンロードしてください"
echo ""