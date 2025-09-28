# Multi-stage build for Runpod Serverless GPU
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    HF_HUB_CACHE=/runpod-volume/huggingface-cache \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    git git-lfs wget curl unzip \
    ffmpeg \
    build-essential cmake \
    libsndfile1 libportaudio2 \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy all requirements files first
COPY requirements.txt /app/
COPY gradio_frontend/requirements.txt /app/gradio_frontend/
COPY gpt_sovits_full/requirements.txt /app/gpt_sovits_full/
COPY facefusion/requirements.txt /app/facefusion/

# Create and setup main venv for Wav2lip
RUN python3.10 -m venv /app/venv && \
    . /app/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir onnxruntime-gpu==1.22.0 && \
    pip install --no-cache-dir \
        opencv-python==4.10.0.84 \
        ffmpeg-python==0.2.0 \
        numpy==1.24.3 \
        Pillow==10.4.0 \
        gradio==4.44.0 \
        tqdm==4.66.5 \
        scipy==1.11.4 \
        librosa==0.10.2 \
        gfpgan==1.3.8 \
        facexlib==0.3.0 \
        basicsr==1.4.2 \
        numba==0.60.0 \
        imageio==2.36.1 \
        imageio-ffmpeg==0.5.1 \
        realesrgan==0.3.0

# Create and setup Gradio frontend venv
RUN python3.10 -m venv /app/gradio_venv && \
    . /app/gradio_venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir gradio>=5.0.0 Pillow>=9.0.0 numpy>=1.21.0

# Create and setup GPT-SoVITS venv
RUN python3.10 -m venv /app/sovits_venv && \
    . /app/sovits_venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r /app/gpt_sovits_full/requirements.txt || true

# Create and setup FaceFusion venv
RUN python3.10 -m venv /app/facefusion_venv && \
    . /app/facefusion_venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/facefusion/requirements.txt || true

# Create and setup Llama venv (if needed)
RUN python3.10 -m venv /app/llama_venv && \
    . /app/llama_venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir llama-cpp-python==0.3.2 || true

# Copy application code
COPY . /app/

# Create model directories
RUN mkdir -p /app/checkpoints \
    /app/utils \
    /app/facefusion/.assets/models \
    /app/gpt_sovits_full/pretrained_models \
    /app/output \
    /tmp/gradio_three_stage

# Download Wav2lip models if not present
RUN if [ ! -f "/app/checkpoints/wav2lip_gan.pth" ]; then \
        wget -O /app/checkpoints/wav2lip_gan.pth \
        "https://github.com/Rudrabha/Wav2Lip/releases/download/models/wav2lip_gan.pth" || true; \
    fi && \
    if [ ! -f "/app/checkpoints/wav2lip.pth" ]; then \
        wget -O /app/checkpoints/wav2lip.pth \
        "https://github.com/Rudrabha/Wav2Lip/releases/download/models/wav2lip.pth" || true; \
    fi && \
    if [ ! -f "/app/checkpoints/s3fd.pth" ]; then \
        wget -O /app/checkpoints/s3fd.pth \
        "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" || true; \
    fi && \
    if [ ! -f "/app/checkpoints/mobilenet.pth" ]; then \
        wget -O /app/checkpoints/mobilenet.pth \
        "https://github.com/Rudrabha/Wav2Lip/releases/download/models/mobilenet.pth" || true; \
    fi

# Download GFPGAN model if not present
RUN if [ ! -f "/app/checkpoints/GFPGANv1.4.pth" ]; then \
        wget -O /app/checkpoints/GFPGANv1.4.pth \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" || true; \
    fi

# Set permissions - simplified to avoid hanging
RUN find /app -maxdepth 2 -name "*.py" | xargs -r chmod +x || true

# Expose ports
EXPOSE 7860 7862 7863 7864 7865 7866

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7866/ || exit 1

# Create startup script for Runpod
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting Wav2Lip Multi-venv System on Runpod..."\n\
\n\
# Check if running on Runpod\n\
if [ ! -z "$RUNPOD_POD_ID" ]; then\n\
    echo "Detected Runpod environment: $RUNPOD_POD_ID"\n\
    # Use Runpod volume for models if available\n\
    if [ -d "/runpod-volume" ]; then\n\
        echo "Using Runpod volume for models"\n\
        # Link model directories if they exist in volume\n\
        [ -d "/runpod-volume/models/wav2lip" ] && ln -sf /runpod-volume/models/wav2lip/* /app/checkpoints/ 2>/dev/null || true\n\
        [ -d "/runpod-volume/models/facefusion" ] && ln -sf /runpod-volume/models/facefusion/* /app/facefusion/.assets/models/ 2>/dev/null || true\n\
        [ -d "/runpod-volume/models/sovits" ] && ln -sf /runpod-volume/models/sovits/* /app/gpt_sovits_full/pretrained_models/ 2>/dev/null || true\n\
    fi\n\
fi\n\
\n\
# Start application based on environment variable or default\n\
if [ "$RUN_MODE" = "handler" ]; then\n\
    echo "Starting Runpod handler mode..."\n\
    cd /app && python3 runpod_handler.py\n\
else\n\
    echo "Starting Gradio interface..."\n\
    cd /app/gradio_frontend && \n\
    source /app/gradio_venv/bin/activate && \n\
    python wav2lip_sovits_llama_integrated.py --server-name 0.0.0.0 --server-port ${PORT:-7866}\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"]