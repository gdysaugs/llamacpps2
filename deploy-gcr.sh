#!/bin/bash
# GCR-optimized build and deployment script

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-wav2lip-project}"
SERVICE_NAME="${SERVICE_NAME:-wav2lip}"
REGION="${REGION:-us-central1}"
BUCKET_NAME="${BUCKET_NAME:-${PROJECT_ID}-models}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Building and deploying Wav2lip to Cloud Run GPU"
echo "================================================"
echo "Project: ${PROJECT_ID}"
echo "Service: ${SERVICE_NAME}"
echo "Region: ${REGION}"
echo "Image: ${IMAGE_NAME}"
echo ""

# Check dependencies
echo "🔍 Checking dependencies..."
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI not found"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker not found"
    exit 1
fi

# Set project
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    storage.googleapis.com \
    --quiet

# Build Docker image with size optimizations
echo "🏗️ Building optimized Docker image..."
echo "Using .dockerignore.gcr for aggressive size reduction..."

# Use Cloud Build for faster building (optional)
if [ "$USE_CLOUD_BUILD" = "true" ]; then
    echo "Using Cloud Build..."
    gcloud builds submit \
        --dockerignore-file=.dockerignore.gcr \
        --dockerfile=Dockerfile.gcr \
        --tag ${IMAGE_NAME} \
        .
else
    echo "Building locally..."
    docker build \
        -f Dockerfile.gcr \
        --dockerignore-file=.dockerignore.gcr \
        -t ${IMAGE_NAME} \
        .

    # Push to GCR
    echo "📤 Pushing image to Google Container Registry..."
    docker push ${IMAGE_NAME}
fi

# Show image size
echo "📏 Image size:"
docker images ${IMAGE_NAME} --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" || \
gcloud container images describe ${IMAGE_NAME} --format="value(imageSizeBytes)" | \
awk '{print "Size: " $1/1024/1024/1024 " GB"}'

# Deploy to Cloud Run with GPU
echo "🚀 Deploying to Cloud Run GPU..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --cpu 4 \
    --memory 16Gi \
    --min-instances 0 \
    --max-instances 100 \
    --timeout 3600 \
    --concurrency 10 \
    --port 8080 \
    --set-env-vars "MODEL_BUCKET_NAME=${BUCKET_NAME}" \
    --set-env-vars "PYTHONUNBUFFERED=1" \
    --allow-unauthenticated

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)')

echo ""
echo "✅ Deployment successful!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo ""
echo "📊 Service Configuration:"
echo "  - GPU: NVIDIA L4 (1 unit)"
echo "  - CPU: 4 vCPU"
echo "  - Memory: 16 GiB"
echo "  - Max instances: 100 (unlimited scaling)"
echo "  - Model storage: gs://${BUCKET_NAME}"
echo ""
echo "🔧 Testing deployment:"
echo "curl -X POST ${SERVICE_URL}/predict \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"video_url\": \"test_video.mp4\", \"audio_url\": \"test_audio.wav\"}'"
echo ""
echo "📈 Monitor logs:"
echo "gcloud run logs tail ${SERVICE_NAME} --region ${REGION}"
echo ""
echo "💰 Estimated costs (T4 GPU equivalent):"
echo "  - Base cost: ~\$0.000336/second when running"
echo "  - Idle cost: \$0 (scales to zero)"
echo "  - 100 requests/day (30 sec each): ~\$3/month"