#!/usr/bin/env python3
"""
Simple Wav2Lip Gradio Frontend Application (Fixed Version)
高品質リップシンク動画生成のための簡単なWebインターフェース
"""

import gradio as gr
import os
import sys
import subprocess
import tempfile
import uuid
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# Wav2Lipプロジェクトのルートディレクトリを追加
WAV2LIP_ROOT = Path(__file__).parent.parent
sys.path.append(str(WAV2LIP_ROOT))

def run_wav2lip_subprocess(video_file: str, audio_file: str):
    """Simple Wav2Lip subprocess execution"""

    if not video_file or not audio_file:
        return None, "❌ 動画と音声ファイルをアップロードしてください"

    # Output filename generation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"wav2lip_result_{timestamp}.mp4"
    temp_dir = Path(tempfile.gettempdir()) / "wav2lip_gradio"
    temp_dir.mkdir(exist_ok=True)
    output_path = temp_dir / output_filename

    # Subprocess script path
    wav2lip_script = WAV2LIP_ROOT / "wav2lip_subprocess.py"
    # Use main venv Python instead of Gradio venv
    main_python = WAV2LIP_ROOT / "venv" / "bin" / "python"

    try:
        # Build command with main venv Python
        cmd = [
            str(main_python),
            str(wav2lip_script),
            video_file,
            audio_file,
            "-o", str(output_path),
            "--device", "cuda"
        ]

        # Force GFPGAN with 30% blend ratio
        cmd.extend(["--gfpgan-blend", "0.3"])

        # Execute subprocess
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(WAV2LIP_ROOT)
        )
        end_time = time.time()
        execution_time = end_time - start_time

        # Log message
        log_message = f"""✅ 処理完了！
⏱️ 実行時間: {execution_time:.2f}秒
📄 出力ファイル: {output_filename}
✨ GFPGAN: 有効（固定）
🎚️ ブレンド比率: 30%

📝 詳細ログ:
{result.stdout}
"""

        if output_path.exists():
            return str(output_path), log_message
        else:
            return None, f"❌ 出力ファイルが生成されませんでした\n{log_message}"

    except subprocess.CalledProcessError as e:
        error_message = f"""❌ 処理エラーが発生しました
💥 リターンコード: {e.returncode}

📝 stdout:
{e.stdout}

📝 stderr:
{e.stderr}
"""
        return None, error_message

    except Exception as e:
        return None, f"❌ 予期しないエラーが発生しました: {str(e)}"

# Gradio Interface
def create_simple_interface():
    """Create simple Gradio interface"""

    with gr.Blocks(
        title="Wav2Lip Simple Frontend",
        theme=gr.themes.Default()
    ) as interface:

        gr.Markdown("""
        # 🎬 Wav2Lip Simple Frontend

        高品質リップシンク動画生成システムのシンプル版インターフェース

        ## 使用方法
        1. 動画ファイルと音声ファイルをアップロード
        2. 「生成開始」ボタンをクリック

        **GFPGAN顔補正は自動で30%適用されます**
        """)

        with gr.Row():
            with gr.Column():
                video_input = gr.File(
                    label="📹 動画ファイル",
                    file_types=[".mp4", ".avi", ".mov", ".mkv"],
                    type="filepath"
                )

                audio_input = gr.File(
                    label="🎵 音声ファイル",
                    file_types=[".mp3", ".wav", ".m4a", ".aac"],
                    type="filepath"
                )

                gr.Markdown("""
                ### ⚙️ 処理設定
                ✨ **GFPGAN顔補正**: 有効（固定30%）
                高品質な顔補正で自然な仕上がりを実現
                """)

                process_btn = gr.Button(
                    "🚀 生成開始",
                    variant="primary"
                )

            with gr.Column():
                output_video = gr.Video(label="生成された動画")

                log_output = gr.Textbox(
                    label="処理ログ",
                    lines=10,
                    interactive=False
                )

        # Event handler
        process_btn.click(
            fn=run_wav2lip_subprocess,
            inputs=[video_input, audio_input],
            outputs=[output_video, log_output]
        )

    return interface

def main():
    """Main function"""
    interface = create_simple_interface()

    print("🚀 Wav2Lip Simple Frontend Starting...")
    print(f"📁 Wav2Lip Root: {WAV2LIP_ROOT}")
    print(f"🌐 Server: http://localhost:7862")

    interface.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        debug=True,
        inbrowser=True
    )

if __name__ == "__main__":
    main()