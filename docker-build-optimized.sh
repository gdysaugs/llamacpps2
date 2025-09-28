#!/bin/bash

# Optimized Docker build script for Runpod deployment

set -e

echo "Building optimized Docker image for Runpod..."
echo "================================================"

# Image name and tag
IMAGE_NAME="wav2lip-runpod"
IMAGE_TAG="optimized"
DOCKER_HUB_USER="${DOCKER_HUB_USER:-yourusername}"  # Set your Docker Hub username

# Full image name
FULL_IMAGE_NAME="${DOCKER_HUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

# Check Docker daemon
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker daemon is not running!"
    exit 1
fi

# Build with optimized Dockerfile
echo "Building Docker image with optimizations..."
docker build \
    -f Dockerfile.optimized \
    -t ${IMAGE_NAME}:${IMAGE_TAG} \
    --platform linux/amd64 \
    .

if [ $? -eq 0 ]; then
    echo "Build successful!"

    # Show image size
    echo ""
    echo "Image size:"
    docker images ${IMAGE_NAME}:${IMAGE_TAG} --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"

    # Tag for Docker Hub
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${FULL_IMAGE_NAME}

    echo ""
    echo "To push to Docker Hub:"
    echo "  1. Login: docker login"
    echo "  2. Push: docker push ${FULL_IMAGE_NAME}"

    echo ""
    echo "To test locally:"
    echo "  docker run -p 7866:7866 --gpus all ${IMAGE_NAME}:${IMAGE_TAG}"

    echo ""
    echo "For Runpod deployment:"
    echo "  Use image: ${FULL_IMAGE_NAME}"
    echo "  Set environment variable: RUN_MODE=handler (for serverless)"
    echo "  Mount volume at: /runpod-volume"
else
    echo "Build failed!"
    exit 1
fi