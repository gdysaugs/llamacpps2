#!/usr/bin/env python3
"""
SoVITS-FaceFusion-Wav2Lip 統合Gradioフロントエンド
音声クローン + 顔交換 + リップシンクの3段階統合Webインターフェース
"""

import gradio as gr
import sys
import tempfile
import time
import gc
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# プロジェクトルートを追加
WAV2LIP_ROOT = Path(__file__).parent.parent
sys.path.append(str(WAV2LIP_ROOT))

# 統合モジュールのインポート
try:
    from gradio_frontend.sovits_wav2lip_integration import SOVITSWav2LipIntegration
    from gradio_frontend.facefusion_integration import FaceFusionIntegration
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import integration modules: {e}")
    INTEGRATION_AVAILABLE = False

class ThreeStageIntegratedApp:
    """3段階統合Gradioアプリケーション"""

    def __init__(self):
        """初期化"""
        self.sovits_wav2lip = SOVITSWav2LipIntegration() if INTEGRATION_AVAILABLE else None
        self.facefusion = FaceFusionIntegration() if INTEGRATION_AVAILABLE else None
        self.temp_dir = Path("/tmp/gradio_three_stage")
        self.temp_dir.mkdir(exist_ok=True)

        # サポートするファイル形式
        self.supported_video_formats = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        self.supported_audio_formats = [".mp3", ".wav", ".m4a", ".aac", ".flac"]
        self.supported_image_formats = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    def validate_inputs(
        self,
        video_file: Optional[str],
        reference_audio_file: Optional[str],
        script_text: str,
        source_image_file: Optional[str] = None
    ) -> Tuple[bool, str]:
        """統合入力検証"""
        if not INTEGRATION_AVAILABLE:
            return False, "❌ 統合システムが利用できません"

        # 基本入力検証
        if not video_file:
            return False, "❌ 動画ファイルをアップロードしてください"

        if not reference_audio_file:
            return False, "❌ リファレンス音声ファイルをアップロードしてください"

        if not script_text or not script_text.strip():
            return False, "❌ セリフテキストを入力してください"

        if len(script_text.strip()) < 5:
            return False, "❌ セリフテキストが短すぎます（5文字以上）"

        if len(script_text.strip()) > 500:
            return False, "❌ セリフテキストが長すぎます（500文字以下）"

        # ファイル形式検証
        video_ext = Path(video_file).suffix.lower()
        audio_ext = Path(reference_audio_file).suffix.lower()

        if video_ext not in self.supported_video_formats:
            return False, f"❌ サポートされていない動画形式: {video_ext}"

        if audio_ext not in self.supported_audio_formats:
            return False, f"❌ サポートされていない音声形式: {audio_ext}"

        # ソース画像検証（オプション）
        if source_image_file:
            if not self.facefusion or not self.facefusion.is_available():
                return False, "❌ FaceFusion機能が利用できません"

            image_ext = Path(source_image_file).suffix.lower()
            if image_ext not in self.supported_image_formats:
                return False, f"❌ サポートされていない画像形式: {image_ext}"

        return True, "✅ 統合入力検証OK"

    def process_three_stage_pipeline(
        self,
        video_file,
        reference_audio_file,
        script_text,
        source_image_file,
        use_gfpgan,
        device,
        progress=gr.Progress()
    ):
        """
        3段階統合パイプライン処理

        Args:
            video_file: 動画ファイル
            reference_audio_file: リファレンス音声ファイル
            script_text: セリフテキスト
            source_image_file: ソース画像ファイル（オプション）
            use_gfpgan: GFPGAN使用フラグ
            device: 処理デバイス
            progress: Gradio進捗表示

        Returns:
            Tuple: (出力動画, ログ, 統計情報)
        """
        try:
            if not INTEGRATION_AVAILABLE:
                return None, "❌ 統合システムが利用できません", None

            # 入力検証
            is_valid, message = self.validate_inputs(
                video_file, reference_audio_file, script_text, source_image_file
            )
            if not is_valid:
                return None, message, None

            # 全体統計情報
            pipeline_start_time = time.time()
            total_stats = {}
            log_messages = []

            # Phase 1: SoVITS音声生成
            progress(0.0, "🎵 Phase 1: SoVITS音声クローン生成中...")
            log_messages.append("=" * 50)
            log_messages.append("🎵 Phase 1: SoVITS音声クローン生成開始")
            log_messages.append("=" * 50)

            def sovits_progress(value, desc):
                progress(value * 0.3, f"🎵 Phase 1: {desc}")

            try:
                audio_result = self.sovits_wav2lip.process_sovits_audio_generation(
                    reference_audio_file, script_text, device, sovits_progress
                )

                if not audio_result or not audio_result.get("success"):
                    error_msg = audio_result.get("message", "不明なエラー") if audio_result else "音声生成失敗"
                    log_messages.append(f"❌ Phase 1失敗: {error_msg}")
                    return None, "\n".join(log_messages), None

                generated_audio = audio_result["audio_path"]
                total_stats["phase1"] = audio_result.get("stats", {})
                log_messages.append(f"✅ Phase 1完了: {generated_audio}")

            except Exception as e:
                log_messages.append(f"❌ Phase 1エラー: {str(e)}")
                return None, "\n".join(log_messages), None

            # Phase 2: FaceFusion顔交換（条件付き）
            current_video = video_file

            if source_image_file and self.facefusion.is_available():
                progress(0.3, "🎭 Phase 2: FaceFusion顔交換処理中...")
                log_messages.append("\n" + "=" * 50)
                log_messages.append("🎭 Phase 2: FaceFusion顔交換開始")
                log_messages.append("=" * 50)

                def facefusion_progress(value, desc):
                    progress(0.3 + value * 0.4, f"🎭 Phase 2: {desc}")

                try:
                    success, swapped_video, ff_log, ff_stats = self.facefusion.process_face_swap(
                        source_image_file, video_file, facefusion_progress
                    )

                    if success and swapped_video:
                        current_video = swapped_video
                        total_stats["phase2"] = ff_stats
                        log_messages.append(f"✅ Phase 2完了: {swapped_video}")
                        log_messages.append(ff_log)

                        # メモリクリーンアップ
                        progress(0.7, "🧹 Phase 2: メモリクリーンアップ中...")
                        self.facefusion.cleanup_memory()

                    else:
                        log_messages.append(f"❌ Phase 2失敗: {ff_log}")
                        # FaceFusion失敗時は元動画で続行
                        log_messages.append("⚠️ 元動画でPhase 3に進行します")

                except Exception as e:
                    log_messages.append(f"❌ Phase 2エラー: {str(e)}")
                    log_messages.append("⚠️ 元動画でPhase 3に進行します")

            else:
                log_messages.append("\n🔄 Phase 2: FaceFusion スキップ（ソース画像なし）")

            # Phase 3: Wav2Lip最終処理
            progress(0.7, "💋 Phase 3: Wav2Lipリップシンク処理中...")
            log_messages.append("\n" + "=" * 50)
            log_messages.append("💋 Phase 3: Wav2Lipリップシンク開始")
            log_messages.append("=" * 50)

            def wav2lip_progress(value, desc):
                progress(0.7 + value * 0.25, f"💋 Phase 3: {desc}")

            try:
                final_result = self.sovits_wav2lip.process_wav2lip_lipsync(
                    current_video, generated_audio, use_gfpgan, device, wav2lip_progress
                )

                if not final_result or not final_result.get("success"):
                    error_msg = final_result.get("message", "不明なエラー") if final_result else "リップシンク失敗"
                    log_messages.append(f"❌ Phase 3失敗: {error_msg}")
                    return None, "\n".join(log_messages), None

                final_video = final_result["video_path"]
                total_stats["phase3"] = final_result.get("stats", {})
                log_messages.append(f"✅ Phase 3完了: {final_video}")

            except Exception as e:
                log_messages.append(f"❌ Phase 3エラー: {str(e)}")
                return None, "\n".join(log_messages), None

            # 最終クリーンアップ
            progress(0.95, "🧹 最終クリーンアップ中...")
            gc.collect()

            # 全体統計情報
            total_time = time.time() - pipeline_start_time
            total_stats["total_time"] = total_time
            total_stats["pipeline_stages"] = 2 if not source_image_file else 3

            # 成功メッセージ
            progress(1.0, "✅ 3段階パイプライン完了!")
            log_messages.append("\n" + "=" * 50)
            log_messages.append(f"🎉 3段階パイプライン完了: {total_time:.1f}秒")
            log_messages.append("=" * 50)

            return final_video, "\n".join(log_messages), total_stats

        except Exception as e:
            error_msg = f"❌ パイプライン実行エラー: {str(e)}"
            return None, error_msg, None

    def create_interface(self):
        """Gradioインターフェース作成"""
        with gr.Blocks(
            title="SoVITS-FaceFusion-Wav2Lip 統合システム",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {max-width: 1200px !important}
            .progress-bar {background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1) !important}
            .stage-header {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px; border-radius: 8px; margin: 10px 0;}
            .facefusion-optional {border: 2px dashed #ffa726; padding: 15px; border-radius: 8px; background: #fff3e0;}
            """
        ) as interface:

            # ヘッダー
            gr.HTML("""
            <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h1>🎭🎵💋 SoVITS-FaceFusion-Wav2Lip 統合システム</h1>
                <p>音声クローン生成 → 顔交換 → リップシンクの3段階統合パイプライン</p>
            </div>
            """)

            with gr.Row():
                # 左側：入力セクション
                with gr.Column(scale=1):
                    gr.HTML('<div class="stage-header"><h3>📁 入力ファイル</h3></div>')

                    # 基本入力（必須）
                    video_input = gr.File(
                        label="🎬 動画ファイル（必須）- 対応形式: MP4, AVI, MOV, MKV, WebM",
                        file_types=["video"]
                    )

                    reference_audio_input = gr.File(
                        label="🎤 リファレンス音声（必須）- 対応形式: MP3, WAV, M4A, AAC, FLAC",
                        file_types=["audio"]
                    )

                    script_input = gr.Textbox(
                        label="📝 セリフテキスト（必須）",
                        placeholder="生成したい音声のテキストを入力...",
                        lines=3,
                        max_lines=5
                    )

                    # FaceFusion オプション
                    gr.HTML('<div class="stage-header"><h3>🎭 FaceFusion オプション</h3></div>')

                    with gr.Group(elem_classes=["facefusion-optional"]):
                        gr.HTML("""
                        <div style="text-align: center; margin-bottom: 10px;">
                            <strong>🎭 顔交換オプション</strong><br>
                            <small>ソース画像をアップロードすると顔交換処理を行います</small>
                        </div>
                        """)

                        source_image_input = gr.File(
                            label="🖼️ ソース画像（オプション）- 顔交換用。なしの場合は元動画のまま処理",
                            file_types=["image"]
                        )

                        facefusion_status = gr.HTML()

                    # 処理設定
                    gr.HTML('<div class="stage-header"><h3>⚙️ 処理設定</h3></div>')

                    use_gfpgan = gr.Checkbox(
                        label="✨ GFPGAN顔補正（推奨）",
                        value=True
                    )

                    device = gr.Radio(
                        choices=["cuda", "cpu"],
                        value="cuda",
                        label="💻 処理デバイス - CUDA（推奨）またはCPU"
                    )

                    # 実行ボタン
                    process_btn = gr.Button(
                        "🚀 3段階パイプライン実行",
                        variant="primary",
                        size="lg"
                    )

                # 右側：出力セクション
                with gr.Column(scale=1):
                    gr.HTML('<div class="stage-header"><h3>📊 処理状況・結果</h3></div>')

                    # 進捗表示
                    progress_bar = gr.Progress()

                    # 結果表示
                    output_video = gr.Video(
                        label="🎬 最終出力動画",
                        interactive=False
                    )

                    # ログ表示
                    log_output = gr.Textbox(
                        label="📝 処理ログ",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                        show_copy_button=True
                    )

                    # 統計情報
                    stats_output = gr.JSON(
                        label="📊 処理統計",
                        visible=False
                    )

            # FaceFusion利用可能性チェック
            def check_facefusion_status():
                if not INTEGRATION_AVAILABLE:
                    return "❌ 統合システムが利用できません"
                elif not self.facefusion or not self.facefusion.is_available():
                    return "❌ FaceFusion機能が利用できません（環境をチェックしてください）"
                else:
                    return "✅ FaceFusion機能利用可能"

            # イベントハンドラー
            interface.load(
                fn=check_facefusion_status,
                outputs=facefusion_status
            )

            process_btn.click(
                fn=self.process_three_stage_pipeline,
                inputs=[
                    video_input,
                    reference_audio_input,
                    script_input,
                    source_image_input,
                    use_gfpgan,
                    device
                ],
                outputs=[output_video, log_output, stats_output]
            )

        return interface

def main():
    """メイン実行"""
    print("🚀 SoVITS-FaceFusion-Wav2Lip 統合システム起動中...")

    app = ThreeStageIntegratedApp()
    interface = app.create_interface()

    print("📍 アクセス先: http://localhost:7865")

    interface.launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=False
    )

if __name__ == "__main__":
    main()