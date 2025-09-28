"""
Modal deployment for wav2lip_sovits_llama_integrated.py
GPT-SoVITS + Wav2Lip + LlamaCPP統合版
"""
import modal
from pathlib import Path
import sys

app = modal.App("wav2lip-sovits-llama")

# ボリューム定義（モデルファイル用）
models_volume = modal.Volume.from_name("wav2lip-models-2025", create_if_missing=True)
output_volume = modal.Volume.from_name("wav2lip-outputs", create_if_missing=True)

# Dockerイメージ定義
wav2lip_image = (
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
        "psutil",
        "opencv-python==4.10.0.84",
        "scipy==1.11.4",
        "librosa==0.10.2",
        "onnxruntime-gpu==1.19.2",
        "requests",
        "llama-cpp-python==0.3.2",
        "tqdm"
    ])
    # PyTorchを別途インストール
    .pip_install([
        "torch==2.4.1+cu121",
        "torchvision==0.19.1+cu121",
        "torchaudio==2.4.1+cu121",
    ], index_url="https://download.pytorch.org/whl/cu121")
    # ファイルコピー
    .copy_local_file(
        "./gradio_frontend/wav2lip_sovits_llama_integrated.py",
        "/app/wav2lip_sovits_llama_integrated.py"
    )
    .copy_local_file(
        "./wav2lip_subprocess.py",
        "/app/wav2lip_subprocess.py"
    )
    .copy_local_file(
        "./sovits_wav2lip_integration.py",
        "/app/sovits_wav2lip_integration.py"
    )
    .copy_local_file(
        "./utils",
        "/app/utils"
    )
    .run_commands([
        "mkdir -p /app/output",
        "mkdir -p /app/temp"
    ])
)

@app.function(
    image=wav2lip_image,
    gpu="T4",
    memory=16384,  # 16GB RAM (LlamaCPP用に増量)
    timeout=1800,
    volumes={
        "/app/models": models_volume,
        "/app/output": output_volume,
    },
    min_containers=1,
)
@modal.concurrent(max_inputs=3)
def run_wav2lip_sovits_llama():
    """
    wav2lip_sovits_llama_integrated.pyを実行
    """
    import sys
    import os
    import subprocess

    # パス設定
    sys.path.append('/app')
    os.chdir('/app')

    # 環境変数設定
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONUNBUFFERED'] = '1'

    # Gradioアプリ実行
    subprocess.run([
        sys.executable,
        "/app/wav2lip_sovits_llama_integrated.py"
    ], check=True)

@app.local_entrypoint()
def main():
    """
    エントリーポイント
    """
    print("🚀 Starting Wav2Lip-SoVITS-Llama on Modal...")
    run_wav2lip_sovits_llama.remote()

if __name__ == "__main__":
    # デプロイ用スクリプト
    print("Deploy with: modal deploy modal_wav2lip_app.py")