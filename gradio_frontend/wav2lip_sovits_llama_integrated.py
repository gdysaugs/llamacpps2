#!/usr/bin/env python3
"""
SOVITS-Wav2Lip-LlamaCPP 統合Gradioフロントエンド（ポータブル配布対応版）
ボイスクローン音声生成 + 口パク動画生成 + AI会話生成の統合Webインターフェース
"""
import gradio as gr
import sys
import tempfile
import time
import gc
import os
import platform
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# ポータブル環境対応のパス設定
def get_app_root():
    """アプリケーションのルートディレクトリを取得（ポータブル対応）"""
    if getattr(sys, 'frozen', False):
        # PyInstaller等で実行ファイル化された場合
        return Path(sys.executable).parent
    else:
        # 通常のPython実行の場合
        return Path(__file__).parent.parent

APP_ROOT = get_app_root()
PYTHON_DIR = APP_ROOT / "python"

# Pythonパス設定
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "gradio_frontend"))

# ポータブルPython実行ファイルのパス
def get_python_executable():
    """ポータブル環境のPython実行ファイルを取得"""
    system = platform.system()

    if system == "Windows":
        python_exe = PYTHON_DIR / "python.exe"
    else:
        python_exe = PYTHON_DIR / "bin" / "python"

    # ポータブルPythonが存在しない場合はシステムのPythonを使用
    if python_exe.exists():
        return str(python_exe)
    else:
        return sys.executable

PYTHON_EXECUTABLE = get_python_executable()

try:
    from gradio_frontend.sovits_wav2lip_integration import SOVITSWav2LipIntegration
    from gradio_frontend.llamacpp_integration import LlamaCPPIntegration
    from gradio_frontend.facefusion_integration import FaceFusionIntegration
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import integration modules: {e}")
    INTEGRATION_AVAILABLE = False

class SOVITSWav2LipLlamaGradioApp:
    def __init__(self):
        """統合Gradioアプリケーションの初期化（ポータブル対応）"""
        self.sovits_integration = SOVITSWav2LipIntegration() if INTEGRATION_AVAILABLE else None
        self.llama_integration = LlamaCPPIntegration() if INTEGRATION_AVAILABLE else None
        self.facefusion_integration = FaceFusionIntegration() if INTEGRATION_AVAILABLE else None

        # ポータブル環境対応の一時ディレクトリ
        self.temp_dir = APP_ROOT / "temp" / "gradio_sovits_wav2lip_llama"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def integrated_pipeline(
        self,
        video_file,
        reference_audio_file,
        script_text,
        use_ai_conversation,
        additional_prompt,
        max_tokens,
        sovits_speed,
        source_image,
        progress=gr.Progress()
    ):
        """統合パイプライン実行（元のシグネチャに合わせる）"""
        if not INTEGRATION_AVAILABLE:
            return None, "❌ 統合システムが利用できません"

        try:
            log_messages = []

            # 入力検証（簡易版）
            if not video_file:
                return None, "❌ 動画ファイルをアップロードしてください"
            if not reference_audio_file:
                return None, "❌ リファレンス音声ファイルをアップロードしてください"
            if not script_text or not script_text.strip():
                return None, "❌ セリフテキストを入力してください"

            # 設定
            use_facefusion = source_image is not None
            use_gfpgan = not use_facefusion
            device = "cuda"
            actual_script_text = script_text

            log_messages.append("🚀 ポータブル版統合パイプライン開始")
            log_messages.append(f"📁 アプリルート: {APP_ROOT}")
            log_messages.append(f"🎭 FaceFusion: {'有効' if use_facefusion else '無効'}")
            log_messages.append(f"✨ GFPGAN: {'有効' if use_gfpgan else '無効'}")

            # AI会話機能（有効な場合）
            if use_ai_conversation and self.llama_integration:
                progress(0.1, "🤖 AI会話生成中...")
                log_messages.append("=" * 50)
                log_messages.append("🤖 AI会話生成開始")
                log_messages.append(f"🔧 max_tokens設定: {max_tokens} (型: {type(max_tokens)})")
                log_messages.append(f"🔧 use_ai_conversation: {use_ai_conversation}")
                log_messages.append(f"🔧 additional_prompt: '{additional_prompt}'")

                try:
                    llama_result = self.llama_integration.generate_response(
                        user_input=script_text,
                        additional_prompt=additional_prompt,
                        max_tokens=int(max_tokens),
                        temperature=0.7
                    )

                    if llama_result and llama_result.get("success"):
                        actual_script_text = llama_result["response"]
                        log_messages.append(f"✅ AI応答生成成功: {actual_script_text[:100]}...")
                    else:
                        error_msg = llama_result.get("message", "不明なエラー") if llama_result else "レスポンスなし"
                        log_messages.append(f"⚠️ AI会話失敗: {error_msg}")
                        log_messages.append("📝 元テキストを使用")

                except Exception as e:
                    import traceback
                    log_messages.append(f"⚠️ AI会話例外: {str(e)}")
                    log_messages.append(f"📋 トレースバック: {traceback.format_exc()[:200]}")

            # Phase 1: SoVITS音声生成
            phase_offset = 0.2 if use_ai_conversation else 0.0
            progress(phase_offset + 0.3, "🎵 Phase 1: SoVITS音声生成中...")
            log_messages.append("=" * 50)
            log_messages.append("🎵 Phase 1: SoVITS音声生成開始")

            try:
                if self.sovits_integration:
                    audio_result = self.sovits_integration.process_sovits_audio_generation(
                        reference_audio_file,
                        actual_script_text,
                        device,
                        speed_factor=sovits_speed
                    )

                    if audio_result and audio_result.get("success"):
                        generated_audio_path = audio_result["audio_path"]
                        log_messages.append(f"✅ Phase 1完了: {generated_audio_path}")
                    else:
                        error_msg = audio_result.get("message", "音声生成失敗") if audio_result else "音声生成失敗"
                        log_messages.append(f"❌ Phase 1失敗: {error_msg}")
                        return None, "\n".join(log_messages)
                else:
                    log_messages.append("❌ SoVITS統合機能が利用できません")
                    return None, "\n".join(log_messages)

            except Exception as e:
                log_messages.append(f"❌ Phase 1エラー: {str(e)}")
                return None, "\n".join(log_messages)

            # Phase 2: Wav2Lip処理
            progress(phase_offset + 0.6, "🎬 Phase 2: Wav2Lip リップシンク中...")
            log_messages.append("=" * 50)
            log_messages.append("🎬 Phase 2: Wav2Lip リップシンク開始")

            try:
                wav2lip_result = self.sovits_integration.process_wav2lip_sync(
                    video_file,
                    generated_audio_path,
                    use_gfpgan,
                    device
                )

                if wav2lip_result and wav2lip_result.get("success"):
                    output_video_path = wav2lip_result["video_path"]
                    log_messages.append(f"✅ Phase 2完了: {output_video_path}")
                else:
                    error_msg = wav2lip_result.get("message", "リップシンク失敗") if wav2lip_result else "リップシンク失敗"
                    log_messages.append(f"❌ Phase 2失敗: {error_msg}")
                    return None, "\n".join(log_messages)

            except Exception as e:
                log_messages.append(f"❌ Phase 2エラー: {str(e)}")
                return None, "\n".join(log_messages)

            # Phase 3: FaceFusion処理（オプション）
            final_output_path = output_video_path
            if use_facefusion and self.facefusion_integration:
                progress(phase_offset + 0.9, "🎭 Phase 3: FaceFusion 顔交換中...")
                log_messages.append("=" * 50)
                log_messages.append("🎭 Phase 3: FaceFusion 顔交換開始")

                try:
                    facefusion_result = self.facefusion_integration.process_face_swap_with_gfpgan(
                        source_image,
                        output_video_path
                    )

                    if facefusion_result and facefusion_result.get("success"):
                        final_output_path = facefusion_result["video_path"]
                        log_messages.append(f"✅ Phase 3完了: {final_output_path}")
                    else:
                        log_messages.append("⚠️ Phase 3失敗、Phase 2結果を使用")

                except Exception as e:
                    log_messages.append(f"⚠️ Phase 3エラー: {str(e)}")

            progress(1.0, "🎉 統合パイプライン完了!")
            log_messages.append("=" * 50)
            log_messages.append("🎉 ポータブル版統合パイプライン完了!")
            log_messages.append(f"📁 出力: {final_output_path}")

            return final_output_path, "\n".join(log_messages)

        except Exception as e:
            return None, f"❌ 統合パイプライン処理中にエラー: {str(e)}"

    def cleanup_existing_processes(self):
        """既存のプロセスクリーンアップ（強制終了対応）"""
        print("🔍 DEBUG: cleanup_existing_processes() 呼び出された")
        try:
            cleanup_log = []
            cleanup_log.append("🧹 既存プロセスクリーンアップ開始...")
            cleanup_log.append("🔍 DEBUG: ポータブル版クリーンアップ実行中")

            # 現在のプロセスIDを取得（自分自身を終了しないため）
            import os
            import subprocess
            current_pid = os.getpid()
            cleanup_log.append(f"ℹ️ 現在のプロセスID: {current_pid}")

            # 他のAI関連プロセスを強制終了（サブプロセスのみ）
            target_processes = [
                "test_llama_cli.py",
                "wav2lip_subprocess.py",
                "gpt_sovits_simple_cli.py"
            ]

            killed_count = 0
            for process_name in target_processes:
                try:
                    # psコマンドで該当プロセスを検索し、現在のプロセスを除外して強制終了
                    result = subprocess.run(
                        ["ps", "aux"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )

                    if result.returncode == 0:
                        lines = result.stdout.split('\n')
                        for line in lines:
                            if process_name in line and str(current_pid) not in line:
                                # プロセスIDを抽出
                                parts = line.split()
                                if len(parts) >= 2:
                                    try:
                                        pid = int(parts[1])
                                        # 自分自身と親プロセスでないことを確認
                                        if pid != current_pid and pid != os.getppid():
                                            # 強制終了 (SIGKILL)
                                            subprocess.run(["kill", "-9", str(pid)], timeout=5)
                                            killed_count += 1
                                            cleanup_log.append(f"🔪 プロセス強制終了: {process_name} (PID: {pid})")
                                    except (ValueError, subprocess.SubprocessError) as e:
                                        cleanup_log.append(f"⚠️ プロセス終了失敗: PID {pid} - {e}")

                except Exception as e:
                    cleanup_log.append(f"⚠️ プロセス検索エラー: {process_name} - {e}")

            cleanup_log.append(f"🔪 終了したプロセス数: {killed_count}")

            # GPU メモリクリーンアップ
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    cleanup_log.append("🔥 GPU メモリキャッシュクリア完了")
            except ImportError:
                cleanup_log.append("ℹ️ PyTorch未インストールのためGPUキャッシュクリアスキップ")
            except Exception as e:
                cleanup_log.append(f"⚠️ GPU メモリクリアエラー: {e}")

            # システムメモリクリーンアップ
            gc.collect()
            cleanup_log.append("🧹 システムメモリクリーンアップ完了")

            # 一時ファイルクリーンアップ
            try:
                temp_dirs = [
                    "/tmp/sovits_wav2lip_integration",
                    "/tmp/gradio_sovits_wav2lip_llama",
                    str(self.temp_dir)  # ポータブル版の一時ディレクトリも含める
                ]
                for temp_dir in temp_dirs:
                    temp_path = Path(temp_dir)
                    if temp_path.exists():
                        import shutil
                        shutil.rmtree(temp_path, ignore_errors=True)
                        cleanup_log.append(f"🗑️ 一時ディレクトリクリア: {temp_dir}")
            except Exception as e:
                cleanup_log.append(f"⚠️ 一時ファイルクリアエラー: {e}")

            cleanup_log.append("✅ 自動プロセス終了・クリーンアップ完了")
            return "\n".join(cleanup_log)

        except Exception as e:
            return f"❌ プロセスクリーンアップエラー: {str(e)}"

    def create_interface(self):
        """Gradio Blocksインターフェース作成（元のレイアウトに準拠）"""

        with gr.Blocks(title="🎭 AI音声合成・口パク・顔交換統合システム（ポータブル版）") as interface:
            gr.Markdown("# 🎭 AI音声合成・口パク・顔交換統合システム（ポータブル版）")
            gr.Markdown("音声クローン + 口パク生成 + AI会話 + FaceFusion顔交換（オプション）")

            with gr.Row():
                with gr.Column(scale=1):
                    video_file = gr.File(
                        label="🎬 動画ファイル",
                        file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                        file_count="single"
                    )
                    reference_audio = gr.File(
                        label="🎵 リファレンス音声",
                        file_types=[".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"],
                        file_count="single"
                    )
                    script_text = gr.Textbox(label="📝 セリフテキスト", lines=4)
                    use_ai_conversation = gr.Checkbox(label="🤖 AI会話モード")
                    additional_prompt = gr.Textbox(label="🎯 キャラクター特徴", lines=2)
                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=1000,
                        value=200,
                        step=10,
                        label="🤖 AI応答文字数"
                    )
                    sovits_speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="🎵 SoVITS音声速度"
                    )
                    source_image = gr.File(
                        label="🎭 FaceFusionソース画像（オプション）",
                        file_types=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
                        file_count="single"
                    )

                with gr.Column(scale=1):
                    output_video = gr.Video(label="🎬 生成された動画")
                    process_log = gr.Textbox(label="📊 処理ログ", lines=20, interactive=False)

            with gr.Row():
                generate_btn = gr.Button("🚀 生成開始", variant="primary")
                cleanup_btn = gr.Button("🧹 プロセスクリーンアップ", variant="secondary")

            # イベントハンドラ
            generate_btn.click(
                fn=self.integrated_pipeline,
                inputs=[
                    video_file,
                    reference_audio,
                    script_text,
                    use_ai_conversation,
                    additional_prompt,
                    max_tokens,
                    sovits_speed,
                    source_image
                ],
                outputs=[output_video, process_log]
            )

            # クリーンアップボタンの処理
            cleanup_btn.click(
                fn=self.cleanup_existing_processes,
                inputs=[],
                outputs=[process_log]
            )

            # ポータブル版情報表示
            with gr.Row():
                gr.Markdown(f"""
                ---
                **ポータブル版情報** | アプリルート: `{APP_ROOT}` | Python: `{PYTHON_EXECUTABLE}` | 統合機能: {"✅" if INTEGRATION_AVAILABLE else "❌"}
                """)

        return interface

def main():
    """メイン関数"""
    print(f"🚀 SoVITS-Wav2Lip-LlamaCPP統合システム（ポータブル版）を開始...")
    print(f"📁 アプリルート: {APP_ROOT}")
    print(f"🐍 Python実行ファイル: {PYTHON_EXECUTABLE}")
    print(f"🔧 統合機能: {'利用可能' if INTEGRATION_AVAILABLE else '利用不可'}")

    app_instance = SOVITSWav2LipLlamaGradioApp()
    interface = app_instance.create_interface()

    # ポートを8080に設定（Docker環境用）
    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=True,
        debug=False,
        allowed_paths=[str(APP_ROOT / "output"), str(APP_ROOT / "temp")]
    )

if __name__ == "__main__":
    main()