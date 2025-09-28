#!/usr/bin/env python3
"""
Wav2Lip Gradio Frontend Application
高品質リップシンク動画生成のためのWebインターフェース
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
from typing import Optional, Tuple, Dict, Any

# Wav2Lipプロジェクトのルートディレクトリを追加
WAV2LIP_ROOT = Path(__file__).parent.parent
sys.path.append(str(WAV2LIP_ROOT))

class Wav2LipGradioApp:
    def __init__(self):
        """Gradioアプリケーションの初期化"""
        self.wav2lip_root = WAV2LIP_ROOT
        self.wav2lip_script = self.wav2lip_root / "wav2lip_subprocess.py"
        self.temp_dir = Path(tempfile.gettempdir()) / "wav2lip_gradio"
        self.temp_dir.mkdir(exist_ok=True)

        # サポートするファイル形式
        self.supported_video_formats = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        self.supported_audio_formats = [".mp3", ".wav", ".m4a", ".aac", ".flac"]

    def validate_files(self, video_file: Optional[str], audio_file: Optional[str]) -> Tuple[bool, str]:
        """アップロードされたファイルの検証"""
        if not video_file:
            return False, "動画ファイルをアップロードしてください"

        if not audio_file:
            return False, "音声ファイルをアップロードしてください"

        # ファイル形式の検証
        video_ext = Path(video_file).suffix.lower()
        audio_ext = Path(audio_file).suffix.lower()

        if video_ext not in self.supported_video_formats:
            return False, f"サポートされていない動画形式です: {video_ext}"

        if audio_ext not in self.supported_audio_formats:
            return False, f"サポートされていない音声形式です: {audio_ext}"

        return True, "ファイル検証OK"

    def generate_output_filename(self) -> str:
        """出力ファイル名の生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"wav2lip_result_{timestamp}_{unique_id}.mp4"

    def run_wav2lip_subprocess(
        self,
        video_file: str,
        audio_file: str,
        use_gfpgan: bool = True,
        device: str = "cuda"
    ) -> Tuple[Optional[str], str, Dict[str, Any]]:
        """
        Wav2Lipサブプロセスの実行 (onnxruntime-gpu 1.15.1最適化版)

        Returns:
            Tuple[Optional[str], str, Dict[str, Any]]: (出力ファイルパス, ログメッセージ, 統計情報)
        """
        try:
            # 出力ファイルパスの生成
            output_filename = self.generate_output_filename()
            output_path = self.temp_dir / output_filename

            # コマンドライン引数の構築（新しいサブプロセス版対応）
            cmd = [
                sys.executable,
                str(self.wav2lip_script),
                video_file,
                audio_file,
                "-o", str(output_path),
                "--device", device
            ]

            # GFPGAN オプション（シンプル化）
            if not use_gfpgan:
                cmd.append("--no-gfpgan")

            # サブプロセス実行
            start_time = time.time()

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(self.wav2lip_root)
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # 統計情報の生成（新しいサブプロセス版対応）
            stats = {
                "success": True,
                "execution_time": execution_time,
                "output_file": str(output_path),
                "settings": {
                    "use_gfpgan": use_gfpgan,
                    "device": device
                }
            }

            # ログメッセージの生成（シンプル化）
            gfpgan_status = '有効 (GPU加速)' if use_gfpgan else '無効 (高速モード)'

            log_message = f"""✅ 処理完了！ (onnxruntime-gpu 1.15.1最適化版)
⏱️ 実行時間: {execution_time:.2f}秒
📄 出力ファイル: {output_filename}
📊 設定:
  - GFPGAN顔補正: {gfpgan_status}
  - 処理デバイス: {device.upper()}
  - RetinaFace検出: GPU加速
  - 元解像度保持: 有効

🚀 パフォーマンス:
{result.stdout.split('Face detection:')[1].split('Inference:')[0] if 'Face detection:' in result.stdout else ''}
{result.stdout.split('Inference:')[1].split('ffmpeg')[0] if 'Inference:' in result.stdout else ''}

📝 詳細ログ:
{result.stdout}
"""

            if output_path.exists():
                # Return absolute path for Gradio
                abs_output_path = output_path.resolve()
                return str(abs_output_path), log_message, stats
            else:
                return None, f"❌ 出力ファイルが生成されませんでした\n{log_message}", stats

        except subprocess.CalledProcessError as e:
            error_message = f"""❌ 処理エラーが発生しました
💥 リターンコード: {e.returncode}

📝 stdout:
{e.stdout}

📝 stderr:
{e.stderr}
"""
            stats = {"success": False, "error": str(e)}
            return None, error_message, stats

        except Exception as e:
            error_message = f"❌ 予期しないエラーが発生しました: {str(e)}"
            stats = {"success": False, "error": str(e)}
            return None, error_message, stats

    def process_video(
        self,
        video_file,
        audio_file,
        use_gfpgan,
        device,
        progress=gr.Progress()
    ):
        """
        メイン処理関数（Gradioインターフェース用 - onnxruntime-gpu 1.15.1対応）
        """
        try:
            progress(0.1, desc="ファイル検証中...")

            # ファイル検証
            is_valid, validation_message = self.validate_files(video_file, audio_file)
            if not is_valid:
                return None, f"❌ {validation_message}", None

            progress(0.2, desc="RetinaFace GPU検出・Wav2Lip推論開始...")

            # Wav2Lip実行（シンプル化されたパラメータ）
            output_file, log_message, stats = self.run_wav2lip_subprocess(
                video_file,
                audio_file,
                use_gfpgan,
                device
            )

            progress(1.0, desc="処理完了！GPU加速で高速処理されました")

            return output_file, log_message, stats

        except Exception as e:
            error_msg = f"❌ 処理中にエラーが発生しました: {str(e)}"
            return None, error_msg, None

    def create_interface(self):
        """Gradioインターフェースの作成"""

        # カスタムCSS
        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .output-video {
            max-height: 500px;
        }
        .log-output {
            max-height: 400px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
        }
        """

        with gr.Blocks(
            title="Wav2Lip ONNX High Quality - Web Interface",
            css=custom_css,
            theme=gr.themes.Soft()
        ) as interface:

            # ヘッダー
            gr.Markdown("""
            # 🎬 Wav2Lip ONNX GPU - Web Interface (v2.0)

            onnxruntime-gpu 1.15.1最適化版。RetinaFace GPU検出 + GFPGAN GPU加速による高品質リップシンク。

            ## ⚡ パフォーマンス
            - **Face detection**: ~76 it/s (RetinaFace GPU)
            - **Inference**: ~6.3 it/s (Wav2Lip GPU)
            - **処理時間**: 約31秒 (5秒動画、GFPGAN有効)

            ## 📋 使用方法
            1. 動画ファイルと音声ファイルをアップロード
            2. GFPGAN設定とデバイス選択
            3. 「🚀 リップシンク生成開始」ボタンをクリック
            4. GPU加速で高速処理、完了後ダウンロード
            """)

            with gr.Row():
                # 左側: 入力とオプション
                with gr.Column(scale=1):
                    gr.Markdown("## 📁 ファイルアップロード")

                    video_input = gr.File(
                        label="📹 動画ファイル",
                        file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                        type="filepath"
                    )

                    audio_input = gr.File(
                        label="🎵 音声ファイル",
                        file_types=[".mp3", ".wav", ".m4a", ".aac", ".flac"],
                        type="filepath"
                    )

                    gr.Markdown("## ⚙️ 処理オプション")

                    with gr.Group():
                        use_gfpgan = gr.Checkbox(
                            label="✨ GFPGAN顔補正を使用",
                            value=True,
                            info="顔画質向上（処理時間が長くなります）"
                        )


                        device = gr.Radio(
                            label="💻 処理デバイス",
                            choices=["cuda", "cpu"],
                            value="cuda",
                            info="CUDAが利用可能な場合はcuda推奨"
                        )

                    # 処理開始ボタン
                    process_btn = gr.Button(
                        "🚀 リップシンク生成開始",
                        variant="primary",
                        size="lg"
                    )

                # 右側: 出力と結果
                with gr.Column(scale=1):
                    gr.Markdown("## 📺 処理結果")

                    output_video = gr.Video(
                        label="生成された動画",
                        elem_classes=["output-video"]
                    )

                    download_btn = gr.DownloadButton(
                        label="💾 動画をダウンロード",
                        variant="secondary",
                        visible=False
                    )

                    gr.Markdown("## 📊 処理ログ")

                    log_output = gr.Textbox(
                        label="ログ・統計情報",
                        lines=15,
                        max_lines=20,
                        elem_classes=["log-output"],
                        interactive=False
                    )

            # イベントハンドラー
            def on_process_click(*args):
                result = self.process_video(*args)
                output_file, log_message, stats = result

                if output_file and Path(output_file).exists():
                    return (
                        output_file,  # output_video
                        log_message,  # log_output
                        gr.update(visible=True, value=output_file)  # download_btn
                    )
                else:
                    return (
                        None,  # output_video
                        log_message,  # log_output
                        gr.update(visible=False)  # download_btn
                    )

            process_btn.click(
                fn=on_process_click,
                inputs=[
                    video_input,
                    audio_input,
                    use_gfpgan,
                    device
                ],
                outputs=[
                    output_video,
                    log_output,
                    download_btn
                ]
            )

            # フッター
            gr.Markdown("""
            ---

            **🔧 開発情報**
            - **バージョン**: v1.3 - サブプロセス対応版
            - **開発者**: Claude Code Assistant
            - **更新日**: 2025-09-13

            **📝 注意事項**
            - 大きなファイルや長時間の動画は処理に時間がかかります
            - GFPGAN使用時は処理時間が大幅に増加します
            - CUDA対応GPUが推奨されます
            """)

        return interface

    def launch(self, **kwargs):
        """Gradioアプリケーションの起動"""
        interface = self.create_interface()

        # デフォルト設定
        launch_kwargs = {
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "share": False,
            "debug": True,
            "inbrowser": True
        }
        launch_kwargs.update(kwargs)

        print("🚀 Wav2Lip Gradio Frontend Starting...")
        print(f"📁 Wav2Lip Root: {self.wav2lip_root}")
        print(f"📁 Temp Directory: {self.temp_dir}")
        print(f"🌐 Server: http://localhost:{launch_kwargs['server_port']}")

        # Add allowed_paths for output directory (Gradio 5.x compatibility)
        launch_kwargs["allowed_paths"] = [str(self.wav2lip_root / "output")]

        interface.launch(**launch_kwargs)

def main():
    """メイン関数"""
    app = Wav2LipGradioApp()
    app.launch()

if __name__ == "__main__":
    main()