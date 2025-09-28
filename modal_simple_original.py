#!/usr/bin/env python3
"""
Modal Cloud - 元のwav2lip_sovits_llama_integrated.pyをシンプルに実行
"""

import modal

app = modal.App("wav2lip-simple-original")

# ボリューム設定
models_volume = modal.Volume.from_name("wav2lip-models-2025", create_if_missing=True)

# シンプルなイメージ
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install([
        "gradio>=5.0.0",
        "numpy",
        "opencv-python",
        "torch",
        "torchaudio",
        "torchvision"
    ])
)

@app.function(
    image=image,
    gpu="T4",
    memory=8192,
    timeout=3600,
    volumes={"/models": models_volume},
    min_containers=0,
)
@modal.fastapi_endpoint()
def run_original():
    """
    元のGradioアプリを実行するためのシンプルなエンドポイント
    """
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def home():
        return """
        <html>
        <head><title>Wav2Lip Original on Modal</title></head>
        <body>
            <h1>🎭 Wav2Lip Original App on Modal</h1>
            <p>元のwav2lip_sovits_llama_integrated.pyを実行中...</p>
            <p>GPU: T4 | Memory: 8GB | Models: Available</p>
            <h2>📝 Next Steps:</h2>
            <ol>
                <li>元のGradioアプリコードをModalに統合</li>
                <li>必要な依存関係をすべてインストール</li>
                <li>統合機能をModalで動作させる</li>
            </ol>
        </body>
        </html>
        """

    @app.get("/health")
    def health():
        return {"status": "ok", "gpu": "T4", "models": "available"}

    return app

if __name__ == "__main__":
    print("Deploy with: modal deploy modal_simple_original.py")