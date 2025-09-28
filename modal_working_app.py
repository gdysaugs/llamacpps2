#!/usr/bin/env python3
"""
確実に動作するModal Gradioアプリケーション
"""

import modal

app = modal.App("wav2lip-working")

# ボリューム設定
models_volume = modal.Volume.from_name("wav2lip-models-2025", create_if_missing=True)

# 軽量イメージ定義
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["ffmpeg"])
    .pip_install([
        "gradio==4.42.0",  # 安定版を使用
        "numpy",
        "requests"
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
def gradio_interface():
    """
    純粋なGradioアプリケーション（FastAPI統合なし）
    """
    import gradio as gr
    import subprocess
    import tempfile
    import os

    def process_files(video_file, audio_file):
        """
        動画と音声を合成する関数
        """
        if not video_file or not audio_file:
            return None, "❌ 動画と音声の両方をアップロードしてください"

        try:
            # 出力ファイル作成
            output_file = tempfile.mktemp(suffix='.mp4')

            # FFmpegコマンド実行
            cmd = [
                'ffmpeg', '-i', video_file, '-i', audio_file,
                '-c:v', 'copy', '-c:a', 'aac',
                '-map', '0:v:0', '-map', '1:a:0',
                '-shortest', '-y', output_file
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0 and os.path.exists(output_file):
                return output_file, "✅ 処理完了！動画と音声を合成しました"
            else:
                return None, f"❌ エラー: {result.stderr}"

        except subprocess.TimeoutExpired:
            return None, "❌ タイムアウト: 処理に時間がかかりすぎました"
        except Exception as e:
            return None, f"❌ エラー: {str(e)}"

    # Gradioインターフェース作成
    interface = gr.Interface(
        fn=process_files,
        inputs=[
            gr.Video(label="🎬 動画ファイル"),
            gr.Audio(label="🎵 音声ファイル", type="filepath")
        ],
        outputs=[
            gr.Video(label="📹 合成結果"),
            gr.Textbox(label="📋 処理状況")
        ],
        title="🎭 Wav2Lip on Modal",
        description="動画ファイルと音声ファイルをアップロードして合成します",
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )

    return interface

@app.local_entrypoint()
def main():
    """
    ローカルエントリーポイント
    """
    interface = gradio_interface.remote()
    interface.launch(
        share=True,
        server_port=7860,
        server_name="0.0.0.0"
    )

if __name__ == "__main__":
    print("Deploy with: modal serve modal_working_app.py")