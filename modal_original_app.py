#!/usr/bin/env python3
"""
Modal Cloud - 元のwav2lip_sovits_llama_integrated.pyをそのまま実行
"""

import modal

app = modal.App("wav2lip-original")

# ボリューム設定
models_volume = modal.Volume.from_name("wav2lip-models-2025", create_if_missing=True)

# 完全な依存関係を含むイメージ
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "libsm6",
        "libxext6", "libxrender1", "libfontconfig1", "libice6",
        "libgomp1", "wget", "curl", "procps", "cmake", "build-essential"
    ])
    # 基本パッケージ
    .pip_install([
        "gradio>=5.0.0",
        "numpy==1.24.3",
        "Pillow>=9.0.0",
        "opencv-python==4.10.0.84",
        "scipy==1.11.4",
        "librosa==0.10.2",
        "onnxruntime-gpu==1.19.2",
        "requests",
        "tqdm",
        "psutil"
    ])
    # PyTorch
    .pip_install([
        "torch==2.4.1+cu121",
        "torchvision==0.19.1+cu121",
        "torchaudio==2.4.1+cu121",
    ], index_url="https://download.pytorch.org/whl/cu121")
    # LlamaCPP
    .pip_install([
        "llama-cpp-python==0.3.2"
    ])
    .run_commands([
        "mkdir -p /app/output",
        "mkdir -p /app/temp",
        "mkdir -p /tmp/gradio_sovits_wav2lip_llama"
    ])
    # プロジェクトファイルを追加（copy=Trueで直接コピー）
    .add_local_dir("./gradio_frontend", "/app/gradio_frontend", copy=True)
    .add_local_dir("./utils", "/app/utils", copy=True)
    .add_local_file("./wav2lip_subprocess.py", "/app/wav2lip_subprocess.py", copy=True)
    .add_local_file("./sovits_wav2lip_integration.py", "/app/sovits_wav2lip_integration.py", copy=True)
    .add_local_file("./wav2lip_inference.py", "/app/wav2lip_inference.py", copy=True)
)

@app.function(
    image=image,
    gpu="T4",
    memory=16384,  # 16GB RAM（統合機能のため）
    timeout=3600,
    volumes={
        "/app/models": models_volume,
    },
    min_containers=1,  # 統合アプリのため1台常時起動
)
@modal.asgi_app()
def original_app():
    """
    元のwav2lip_sovits_llama_integrated.pyをそのまま実行
    """
    import sys
    import os
    from pathlib import Path

    # パス設定
    sys.path.insert(0, '/app')
    sys.path.insert(0, '/app/gradio_frontend')
    os.chdir('/app')

    # 環境変数設定
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONUNBUFFERED'] = '1'

    # モデルパス設定（Modalボリューム使用）
    os.environ['WAV2LIP_MODELS_PATH'] = '/app/models'

    # 元のアプリケーションをインポートして実行
    from wav2lip_sovits_llama_integrated import SOVITSWav2LipLlamaGradioApp

    # アプリケーション初期化
    app_instance = SOVITSWav2LipLlamaGradioApp()

    # Gradioインターフェース作成
    interface = app_instance.create_interface()

    # FastAPIアプリとして返す
    return interface.app

if __name__ == "__main__":
    print("Deploy with: modal deploy modal_original_app.py")