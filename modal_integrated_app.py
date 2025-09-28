"""
Modal Cloud デプロイメント - Wav2Lip統合アプリケーション
Gradio UIを含む完全統合版
"""
import modal
from pathlib import Path

app = modal.App("wav2lip-integrated")

# ボリューム定義
models_volume = modal.Volume.from_name("wav2lip-models-2025", create_if_missing=True)
output_volume = modal.Volume.from_name("wav2lip-outputs", create_if_missing=True)

# Dockerイメージ定義
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "libsm6",
        "libxext6", "libxrender1", "libfontconfig1", "libice6",
        "libgomp1", "wget", "curl", "procps"
    ])
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
        "tqdm",
        "imageio==2.34.0",
        "imageio-ffmpeg==0.4.9"
    ])
    .pip_install([
        "torch==2.4.1+cu121",
        "torchvision==0.19.1+cu121",
        "torchaudio==2.4.1+cu121",
    ], index_url="https://download.pytorch.org/whl/cu121")
)

@app.function(
    image=image,
    gpu="T4",
    memory=8192,
    timeout=1800,
    volumes={
        "/app/models": models_volume,
        "/app/output": output_volume,
    },
    min_containers=1,
)
@modal.asgi_app()
def gradio_app():
    """
    Gradio Web UIアプリケーション
    """
    import gradio as gr
    import subprocess
    import tempfile
    import shutil
    import os
    from pathlib import Path
    import gc
    import torch

    # CUDAデバイス設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def process_wav2lip(video_file, audio_file, enable_gfpgan=True):
        """
        Wav2Lip処理を実行
        """
        try:
            if not video_file or not audio_file:
                return None, "⚠️ 動画ファイルと音声ファイルの両方が必要です"

            # 一時ファイル作成
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_out:
                output_path = tmp_out.name

            # Wav2Lip実行コマンド構築
            cmd = [
                "python", "-u", "/app/wav2lip_inference.py",
                "--checkpoint_path", "/app/models/wav2lip_gan.pth",
                "--face", video_file,
                "--audio", audio_file,
                "--outfile", output_path,
                "--resize_factor", "1"
            ]

            if enable_gfpgan:
                cmd.append("--enable_gfpgan")

            # 処理実行
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # リアルタイムログ取得
            logs = []
            for line in process.stdout:
                logs.append(line.strip())
                yield None, "\n".join(logs[-20:])  # 最新20行表示

            process.wait()

            if process.returncode == 0 and os.path.exists(output_path):
                # 出力ディレクトリにコピー
                final_output = f"/app/output/result_{Path(video_file).stem}.mp4"
                shutil.copy2(output_path, final_output)

                # メモリクリーンアップ
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return final_output, "✅ 処理完了！"
            else:
                return None, "❌ 処理中にエラーが発生しました"

        except Exception as e:
            return None, f"❌ エラー: {str(e)}"
        finally:
            # 一時ファイルクリーンアップ
            if 'output_path' in locals() and os.path.exists(output_path):
                os.remove(output_path)

    # Gradio インターフェース作成
    with gr.Blocks(title="Wav2Lip on Modal", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎭 Wav2Lip リップシンク生成")
        gr.Markdown("動画と音声をアップロードして、リップシンク動画を生成します")

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="動画ファイル",
                    sources=["upload"],
                    format="mp4"
                )
                audio_input = gr.Audio(
                    label="音声ファイル",
                    sources=["upload"],
                    type="filepath"
                )

                with gr.Row():
                    gfpgan_checkbox = gr.Checkbox(
                        label="✨ GFPGAN顔補正を有効化",
                        value=True
                    )

                process_btn = gr.Button(
                    "🚀 リップシンク生成開始",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1):
                output_video = gr.Video(
                    label="生成結果",
                    format="mp4",
                    autoplay=True
                )
                status_text = gr.Textbox(
                    label="処理ログ",
                    lines=10,
                    max_lines=20,
                    interactive=False
                )

        # イベントハンドラ
        process_btn.click(
            fn=process_wav2lip,
            inputs=[video_input, audio_input, gfpgan_checkbox],
            outputs=[output_video, status_text]
        )

        # フッター
        gr.Markdown("---")
        gr.Markdown("🎯 Modal Cloud上で動作中 | GPU: T4 | Memory: 8GB")

    return demo.launch(server_name="0.0.0.0", server_port=7860)

# ローカルテスト用
if __name__ == "__main__":
    print("Deploy with: modal deploy modal_integrated_app.py")