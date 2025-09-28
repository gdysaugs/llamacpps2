#!/bin/bash
# 統合アプリ用Cloud Run GPU デプロイスクリプト
# venv分離維持・モデル含む・サイズ最適化版

set -e

# 設定
PROJECT_ID="${PROJECT_ID:-wav2lip-integrated}"
SERVICE_NAME="${SERVICE_NAME:-wav2lip-integrated}"
REGION="${REGION:-us-central1}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 統合Wav2lip+SoVITS+Llama アプリをCloud Run GPUにデプロイ"
echo "============================================================"
echo "Project: ${PROJECT_ID}"
echo "Service: ${SERVICE_NAME}"
echo "Region: ${REGION}"
echo "Image: ${IMAGE_NAME}"
echo ""
echo "構成: Wav2lip + GPT-SoVITS + Llama + FaceFusion 統合"
echo "venv: 分離維持 (依存関係競合回避)"
echo "モデル: コンテナ内包含"
echo "目標サイズ: 50GB → 15-20GB (60-70%削減)"
echo ""

# 依存関係チェック
echo "🔍 依存関係チェック..."
for cmd in gcloud docker; do
    if ! command -v $cmd &> /dev/null; then
        echo "❌ Error: $cmd が見つかりません"
        exit 1
    fi
done

# プロジェクト設定
gcloud config set project ${PROJECT_ID}

# 必要API有効化
echo "🔧 必要APIを有効化..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    --quiet

# Docker イメージ サイズ推定
echo "📏 予想イメージサイズ:"
echo "  - Base CUDA runtime: ~2GB"
echo "  - PyTorch (5 venv): ~8GB"
echo "  - Models: ~3-5GB"
echo "  - Application code: ~1GB"
echo "  - 合計予想: 15-20GB (元50GBから60-70%削減)"
echo ""

# ビルド方式選択
BUILD_METHOD="${BUILD_METHOD:-local}"
if [ "$BUILD_METHOD" = "cloud" ]; then
    echo "☁️ Cloud Buildを使用..."
    # Cloud Buildの場合、タイムアウト延長
    gcloud builds submit \
        --dockerignore-file=.dockerignore.integrated \
        --dockerfile=Dockerfile.gcr-integrated \
        --timeout=3600s \
        --machine-type=e2-highcpu-32 \
        --tag ${IMAGE_NAME} \
        .
else
    echo "🏗️ ローカルビルド（推奨）..."
    echo "大きなイメージのため、ビルド時間: 15-30分程度"

    # ビルド実行
    docker build \
        -f Dockerfile.gcr-integrated \
        --platform linux/amd64 \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        -t ${IMAGE_NAME} \
        .

    # プッシュ
    echo "📤 イメージをGCRにプッシュ..."
    docker push ${IMAGE_NAME}
fi

# イメージサイズ確認
echo "📊 最終イメージサイズ:"
docker images ${IMAGE_NAME} --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" 2>/dev/null || \
gcloud container images describe ${IMAGE_NAME} --format="get(imageSizeBytes)" 2>/dev/null | \
awk '{printf "Size: %.1f GB\n", $1/1024/1024/1024}'

# Cloud Run GPU にデプロイ
echo "🚀 Cloud Run GPU にデプロイ中..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --cpu 4 \
    --memory 32Gi \
    --min-instances 0 \
    --max-instances 100 \
    --timeout 3600 \
    --concurrency 1 \
    --port 8080 \
    --set-env-vars "PYTHONUNBUFFERED=1" \
    --set-env-vars "CUDA_VISIBLE_DEVICES=0" \
    --allow-unauthenticated

# サービス情報取得
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)')

echo ""
echo "✅ デプロイ完了！"
echo "🌐 サービスURL: ${SERVICE_URL}"
echo ""
echo "📋 サービス構成:"
echo "  - GPU: NVIDIA L4 (24GB VRAM)"
echo "  - CPU: 4 vCPU"
echo "  - Memory: 32 GiB"
echo "  - 同時実行数: 1 (GPUメモリ最適化)"
echo "  - 最大インスタンス: 100"
echo "  - スケール: 0まで自動縮小"
echo ""
echo "🧪 統合機能テスト:"
echo "1. Wav2lip: 動画 + 音声 → 口パク動画"
echo "2. SoVITS: テキスト + 参照音声 → ボイスクローン"
echo "3. Llama: プロンプト → AI会話生成"
echo "4. FaceFusion: 顔画像融合"
echo ""
echo "🔗 ブラウザでアクセス: ${SERVICE_URL}"
echo ""
echo "📈 ログ監視:"
echo "gcloud run logs tail ${SERVICE_NAME} --region ${REGION}"
echo ""
echo "💰 概算コスト (L4 GPU):"
echo "  - 実行時: ~\$1.21/時"
echo "  - 待機時: \$0"
echo "  - 月100回利用: ~\$15-30/月"
echo ""
echo "📝 注意:"
echo "- 初回起動: モデルロード時間込みで1-2分"
echo "- GPU推論: 高品質だが処理時間やや長め"
echo "- 同時実行: 1に制限（GPU競合回避）"
echo ""
echo "🎉 統合AI動画生成サービス準備完了！"