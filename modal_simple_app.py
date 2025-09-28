#!/usr/bin/env python3
"""
Modal Cloud シンプル版 - Wav2Lip Gradioアプリケーション
"""

import modal

app = modal.App("wav2lip-simple")

# ボリューム設定
models_volume = modal.Volume.from_name("wav2lip-models-2025", create_if_missing=True)

# Dockerイメージ定義
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0"
    ])
    .pip_install([
        "gradio==4.44.0",
        "numpy==1.24.3",
        "opencv-python==4.10.0.84"
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
@modal.asgi_app()
def fastapi_app():
    """
    シンプルなGradio Web UI
    """
    import gradio as gr
    from gradio.routes import mount_gradio_app
    from fastapi import FastAPI
    import subprocess
    import tempfile
    import os

    # FastAPIアプリ作成
    app = FastAPI()

    def process_video(video_file, audio_file):
        """
        シンプルな動画処理（FFmpegで合成）
        """
        if not video_file or not audio_file:
            return None, "⚠️ 動画と音声をアップロードしてください"

        try:
            # 出力ファイル
            output_file = tempfile.mktemp(suffix='.mp4')

            # FFmpegで動画と音声を合成
            cmd = [
                'ffmpeg', '-i', video_file, '-i', audio_file,
                '-c:v', 'copy', '-c:a', 'aac',
                '-map', '0:v:0', '-map', '1:a:0',
                '-shortest', '-y', output_file
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                return output_file, "✅ 処理完了！"
            else:
                return None, f"❌ エラー: {result.stderr}"

        except Exception as e:
            return None, f"❌ エラー: {str(e)}"

    # Gradioインターフェース
    with gr.Blocks(title="Wav2Lip Simple") as demo:
        gr.Markdown("# 🎭 Wav2Lip シンプル版")

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="動画")
                audio_input = gr.Audio(label="音声", type="filepath")
                btn = gr.Button("処理開始", variant="primary")

            with gr.Column():
                output_video = gr.Video(label="結果")
                status = gr.Textbox(label="ステータス")

        btn.click(
            fn=process_video,
            inputs=[video_input, audio_input],
            outputs=[output_video, status]
        )

    # Gradioアプリをマウント
    app = mount_gradio_app(app, demo, path="/")

    return app

if __name__ == "__main__":
    print("Deploy with: modal deploy modal_simple_app.py")