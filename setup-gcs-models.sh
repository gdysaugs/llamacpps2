#!/bin/bash
# Setup Google Cloud Storage for model externalization

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-wav2lip-project}"
BUCKET_NAME="${BUCKET_NAME:-${PROJECT_ID}-models}"
REGION="${REGION:-us-central1}"

echo "Setting up GCS model storage for Cloud Run GPU..."
echo "Project: ${PROJECT_ID}"
echo "Bucket: ${BUCKET_NAME}"
echo "Region: ${REGION}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

# Set project
echo "Setting GCP project..."
gcloud config set project ${PROJECT_ID}

# Create GCS bucket for models
echo "Creating GCS bucket: ${BUCKET_NAME}..."
if gsutil ls gs://${BUCKET_NAME} 2>/dev/null; then
    echo "Bucket already exists: gs://${BUCKET_NAME}"
else
    gsutil mb -p ${PROJECT_ID} -c STANDARD -l ${REGION} gs://${BUCKET_NAME}
    echo "Bucket created: gs://${BUCKET_NAME}"
fi

# Create models directory structure
echo "Setting up model directory structure..."
gsutil -q stat gs://${BUCKET_NAME}/models/ 2>/dev/null || {
    echo "Creating models/ directory..."
    echo "" | gsutil cp - gs://${BUCKET_NAME}/models/.keep
}

# Download and upload essential models
echo "Downloading and uploading essential models..."

MODELS_DIR="./temp_models"
mkdir -p ${MODELS_DIR}

# Wav2lip GAN model
if ! gsutil -q stat gs://${BUCKET_NAME}/models/wav2lip_gan.pth; then
    echo "Downloading wav2lip_gan.pth..."
    wget -q -O ${MODELS_DIR}/wav2lip_gan.pth \
        "https://github.com/Rudrabha/Wav2Lip/releases/download/models/wav2lip_gan.pth"

    echo "Uploading wav2lip_gan.pth to GCS..."
    gsutil cp ${MODELS_DIR}/wav2lip_gan.pth gs://${BUCKET_NAME}/models/
else
    echo "wav2lip_gan.pth already exists in GCS"
fi

# Face detection model
if ! gsutil -q stat gs://${BUCKET_NAME}/models/s3fd.pth; then
    echo "Downloading s3fd.pth..."
    wget -q -O ${MODELS_DIR}/s3fd.pth \
        "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"

    echo "Uploading s3fd.pth to GCS..."
    gsutil cp ${MODELS_DIR}/s3fd.pth gs://${BUCKET_NAME}/models/
else
    echo "s3fd.pth already exists in GCS"
fi

# Optional: Upload any existing models
if [ -d "./checkpoints" ]; then
    echo "Uploading existing models from checkpoints/..."
    gsutil -m cp -r ./checkpoints/*.pth gs://${BUCKET_NAME}/models/ 2>/dev/null || true
fi

# Clean up
rm -rf ${MODELS_DIR}

# Set appropriate permissions for Cloud Run
echo "Setting bucket permissions for Cloud Run..."
gsutil iam ch allUsers:objectViewer gs://${BUCKET_NAME}

# Verify uploads
echo "Verifying uploaded models..."
gsutil ls -lh gs://${BUCKET_NAME}/models/

echo ""
echo "✅ GCS model storage setup complete!"
echo ""
echo "Bucket URL: gs://${BUCKET_NAME}"
echo "Environment variable for Cloud Run: MODEL_BUCKET_NAME=${BUCKET_NAME}"
echo ""
echo "Next steps:"
echo "1. Build Docker image: docker build -f Dockerfile.gcr -t gcr.io/${PROJECT_ID}/wav2lip ."
echo "2. Deploy to Cloud Run: gcloud run deploy --image gcr.io/${PROJECT_ID}/wav2lip --gpu 1"