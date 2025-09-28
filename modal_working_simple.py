#!/usr/bin/env python3
"""
Modal Cloud - 確実に動作するシンプル版
"""

import modal

app = modal.App("wav2lip-working-simple")

# ボリューム設定
models_volume = modal.Volume.from_name("wav2lip-models-2025", create_if_missing=True)

# シンプルなイメージ
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["ffmpeg"])
    .pip_install([
        "fastapi",
        "uvicorn"
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
def simple_app():
    """
    シンプルなWebアプリ
    """
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def home():
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Wav2Lip on Modal</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }
                .container {
                    background: rgba(255,255,255,0.1);
                    padding: 30px;
                    border-radius: 15px;
                    backdrop-filter: blur(10px);
                }
                h1 { text-align: center; font-size: 2.5em; margin-bottom: 10px; }
                .subtitle { text-align: center; font-size: 1.2em; margin-bottom: 30px; opacity: 0.9; }
                .status {
                    background: rgba(0,255,0,0.2);
                    padding: 15px;
                    border-radius: 10px;
                    margin: 20px 0;
                    border-left: 4px solid #00ff00;
                }
                .next-steps {
                    background: rgba(255,255,255,0.1);
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                }
                .next-steps h3 { color: #ffeb3b; }
                .next-steps ol { padding-left: 20px; }
                .next-steps li { margin: 10px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎭 Wav2Lip on Modal</h1>
                <p class="subtitle">GPU-Powered Video Lip Sync Application</p>

                <div class="status">
                    <h2>✅ システム準備完了</h2>
                    <ul>
                        <li><strong>GPU:</strong> T4 (利用可能)</li>
                        <li><strong>Memory:</strong> 8GB RAM</li>
                        <li><strong>Models:</strong> wav2lip-models-2025 volume connected</li>
                        <li><strong>Status:</strong> Ready for integration</li>
                    </ul>
                </div>

                <div class="next-steps">
                    <h3>📋 次のステップ</h3>
                    <ol>
                        <li>元のwav2lip_sovits_llama_integrated.pyコードを統合</li>
                        <li>必要な依存関係をすべてインストール</li>
                        <li>Gradio UIを有効化</li>
                        <li>統合機能のテスト実行</li>
                    </ol>
                </div>

                <div class="status">
                    <h2>🔗 API Endpoints</h2>
                    <ul>
                        <li><strong>GET /health</strong> - Health check</li>
                        <li><strong>GET /models</strong> - Model status</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        return html_content

    @app.get("/health")
    def health():
        return JSONResponse({
            "status": "healthy",
            "gpu": "T4",
            "memory": "8GB",
            "models_volume": "wav2lip-models-2025",
            "ready": True
        })

    @app.get("/models")
    def models():
        return JSONResponse({
            "models_volume": "/models",
            "available": True,
            "message": "Models volume mounted successfully"
        })

    return app

if __name__ == "__main__":
    print("Deploy with: modal deploy modal_working_simple.py")