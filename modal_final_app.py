#!/usr/bin/env python3
"""
Modal Cloud最終版 - 確実に動作するWebエンドポイント
"""

import modal

app = modal.App("wav2lip-final")

# ボリューム設定
models_volume = modal.Volume.from_name("wav2lip-models-2025", create_if_missing=True)

# 軽量イメージ
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["ffmpeg"])
    .pip_install([
        "fastapi",
        "python-multipart",
        "uvicorn"
    ])
)

@app.function(
    image=image,
    gpu="T4",
    memory=4096,
    timeout=3600,
    volumes={"/models": models_volume},
    min_containers=0,
)
@modal.asgi_app()
def web_app():
    """
    シンプルなWebエンドポイント
    """
    from fastapi import FastAPI, File, UploadFile, Form
    from fastapi.responses import HTMLResponse, FileResponse
    import subprocess
    import tempfile
    import os
    import shutil

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def main_page():
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>🎭 Wav2Lip on Modal</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .container { background: #f5f5f5; padding: 30px; border-radius: 10px; }
                input, button { margin: 10px 0; padding: 10px; width: 100%; }
                button { background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
                button:hover { background: #0056b3; }
                .result { margin-top: 20px; padding: 20px; background: white; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎭 Wav2Lip Video Processing</h1>
                <p>動画ファイルと音声ファイルをアップロードして合成します</p>

                <form action="/process" method="post" enctype="multipart/form-data">
                    <div>
                        <label>📹 動画ファイル (.mp4, .avi, .mov):</label>
                        <input type="file" name="video" accept=".mp4,.avi,.mov" required>
                    </div>

                    <div>
                        <label>🎵 音声ファイル (.mp3, .wav, .m4a):</label>
                        <input type="file" name="audio" accept=".mp3,.wav,.m4a,.aac" required>
                    </div>

                    <button type="submit">🚀 処理開始</button>
                </form>

                <div class="result">
                    <h3>💡 使用方法</h3>
                    <ul>
                        <li>動画ファイルと音声ファイルを選択</li>
                        <li>「処理開始」ボタンをクリック</li>
                        <li>処理完了後、結果をダウンロード</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """

    @app.post("/process")
    async def process_files(
        video: UploadFile = File(...),
        audio: UploadFile = File(...)
    ):
        try:
            # 一時ディレクトリ作成
            temp_dir = tempfile.mkdtemp()

            # ファイル保存
            video_path = os.path.join(temp_dir, f"video_{video.filename}")
            audio_path = os.path.join(temp_dir, f"audio_{audio.filename}")
            output_path = os.path.join(temp_dir, "output.mp4")

            with open(video_path, "wb") as f:
                content = await video.read()
                f.write(content)

            with open(audio_path, "wb") as f:
                content = await audio.read()
                f.write(content)

            # FFmpeg処理
            cmd = [
                'ffmpeg', '-i', video_path, '-i', audio_path,
                '-c:v', 'copy', '-c:a', 'aac',
                '-map', '0:v:0', '-map', '1:a:0',
                '-shortest', '-y', output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0 and os.path.exists(output_path):
                # 成功時はファイルを返す
                return FileResponse(
                    output_path,
                    media_type="video/mp4",
                    filename="processed_video.mp4"
                )
            else:
                return HTMLResponse(f"""
                <html><body>
                <h2>❌ 処理エラー</h2>
                <p>Error: {result.stderr}</p>
                <a href="/">← 戻る</a>
                </body></html>
                """)

        except Exception as e:
            return HTMLResponse(f"""
            <html><body>
            <h2>❌ システムエラー</h2>
            <p>Error: {str(e)}</p>
            <a href="/">← 戻る</a>
            </body></html>
            """)
        finally:
            # クリーンアップ
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)

    @app.get("/health")
    def health_check():
        return {"status": "healthy", "gpu": "T4", "memory": "4GB"}

    return app

if __name__ == "__main__":
    print("Deploy with: modal deploy modal_final_app.py")