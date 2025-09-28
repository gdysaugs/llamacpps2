#!/usr/bin/env python3
"""
SOVITS-Wav2Lip 統合Gradioフロントエンド
ボイスクローン音声生成 + 口パク動画生成の統合Webインターフェース
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

try:
    from gradio_frontend.sovits_wav2lip_integration import SOVITSWav2LipIntegration
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import integration module: {e}")
    INTEGRATION_AVAILABLE = False

class SOVITSWav2LipGradioApp:
    def __init__(self):
        """統合Gradioアプリケーションの初期化"""
        self.integration = SOVITSWav2LipIntegration() if INTEGRATION_AVAILABLE else None
        self.temp_dir = Path("/tmp/gradio_sovits_wav2lip")
        self.temp_dir.mkdir(exist_ok=True)

        # サポートするファイル形式
        self.supported_video_formats = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        self.supported_audio_formats = [".mp3", ".wav", ".m4a", ".aac", ".flac"]


    def validate_inputs(
        self,
        video_file: Optional[str],
        reference_audio_file: Optional[str],
        script_text: str
    ) -> Tuple[bool, str]:
        """入力検証"""
        if not INTEGRATION_AVAILABLE:
            return False, "❌ 統合システムが利用できません"

        if not video_file:
            return False, "❌ 動画ファイルをアップロードしてください"

        if not reference_audio_file:
            return False, "❌ リファレンス音声ファイルをアップロードしてください"

        if not script_text or not script_text.strip():
            return False, "❌ セリフテキストを入力してください"

        # テキスト長チェック
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

        return True, "✅ 入力検証OK"

    def process_integrated_pipeline(
        self,
        video_file,
        reference_audio_file,
        script_text,
        use_gfpgan,
        device,
        progress=gr.Progress()
    ):
        """
        統合パイプライン処理（Gradio用）

        Args:
            video_file: 動画ファイル
            reference_audio_file: リファレンス音声ファイル
            script_text: セリフテキスト
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
            progress(0.01, desc="入力ファイル検証中...")
            is_valid, validation_message = self.validate_inputs(
                video_file, reference_audio_file, script_text
            )

            if not is_valid:
                return None, validation_message, None

            # 統合処理開始
            progress(0.1, desc="🎭 Phase 1: SOVITS音声クローン開始...")

            print("🚀 統合パイプライン開始")
            print(f"📹 Video: {Path(video_file).name}")
            print(f"🎵 Reference: {Path(reference_audio_file).name}")
            print(f"📝 Script: {script_text[:50]}...")

            # 統合パイプライン実行
            pipeline_start_time = time.time()

            result_video, integrated_log, integrated_stats = self.integration.run_integrated_pipeline(
                video_path=video_file,
                reference_audio_path=reference_audio_file,
                script_text=script_text,
                use_gfpgan=use_gfpgan,
                device=device
            )

            pipeline_end_time = time.time()
            total_time = pipeline_end_time - pipeline_start_time

            # 進捗更新
            if integrated_stats.get("phase1_stats", {}).get("success", False):
                progress(0.4, desc="🎭 Phase 1完了! 🎬 Phase 2: Wav2Lip口パク開始...")

            if integrated_stats.get("phase2_stats", {}).get("success", False):
                progress(0.8, desc="🎬 Phase 2完了! 最終処理中...")

            # 完了処理
            progress(1.0, desc="✅ 統合処理完了!")

            # 結果処理
            if integrated_stats.get("pipeline_success", False) and result_video:
                # ファイルサイズ取得
                final_video_path = Path(result_video)
                if final_video_path.exists():
                    file_size_mb = final_video_path.stat().st_size / (1024 * 1024)

                    # サクセスログ
                    success_log = f"""
🎉 SOVITS-Wav2Lip統合処理完了!

📊 処理サマリー:
⏱️ 総処理時間: {total_time:.2f}秒
📝 セリフ: {script_text[:100]}...

🎭 Phase 1 (SOVITS Voice Clone):
⏱️ 処理時間: {integrated_stats['phase1_stats'].get('execution_time', 0):.2f}秒
📦 音声サイズ: {integrated_stats['phase1_stats'].get('output_size_mb', 0):.2f}MB
✨ 品質: High (GPT-SoVITS v4)

🎬 Phase 2 (Wav2Lip Lip Sync):
⏱️ 処理時間: {integrated_stats['phase2_stats'].get('execution_time', 0):.2f}秒
📦 動画サイズ: {file_size_mb:.2f}MB
⚙️ GFPGAN: {'有効' if use_gfpgan else '無効'}
💻 デバイス: {device.upper()}

🎥 最終結果: リファレンス音声の声質で指定セリフを話すリップシンク動画

{integrated_log}
"""

                    return result_video, success_log, integrated_stats
                else:
                    return None, f"❌ 出力ファイルが見つかりません: {result_video}", integrated_stats
            else:
                # エラーログ
                error_log = f"""
❌ 統合処理失敗!

⏱️ 処理時間: {total_time:.2f}秒
📝 セリフ: {script_text[:100]}...

📊 Phase 1 (SOVITS): {'✅成功' if integrated_stats.get('phase1_stats', {}).get('success') else '❌失敗'}
📊 Phase 2 (Wav2Lip): {'✅成功' if integrated_stats.get('phase2_stats', {}).get('success') else '❌失敗'}

📄 詳細ログ:
{integrated_log}
"""
                return None, error_log, integrated_stats

        except Exception as e:
            error_msg = f"❌ 統合処理中にエラーが発生: {str(e)}"
            print(f"Error in integrated pipeline: {e}")
            return None, error_msg, {"success": False, "error": str(e)}

    def create_interface(self):
        """統合Gradioインターフェースの作成"""

        # カスタムCSS
        custom_css = """
        .gradio-container {
            max-width: 1400px !important;
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
        .script-input {
            min-height: 100px;
        }
        .integration-header {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }
        """

        with gr.Blocks(
            title="SOVITS-Wav2Lip Integration - Voice Clone + Lip Sync",
            css=custom_css,
            theme=gr.themes.Soft()
        ) as interface:

            # ヘッダー
            gr.HTML("""
            <div class="integration-header">
                <h1>🎭🎬 SOVITS-Wav2Lip Integration System</h1>
                <h3>ボイスクローン音声生成 + 口パク動画生成 統合システム</h3>
                <p>リファレンス音声の声質で任意のセリフを話すリップシンク動画を生成</p>
            </div>
            """)

            gr.Markdown("""
            ## 🚀 システム概要

            **統合処理フロー:**
            1. 🎭 **Phase 1**: SOVITS でリファレンス音声の声質を学習し、入力セリフでボイスクローン音声を生成
            2. 🧹 **Memory Cleanup**: Phase 1完了後、メモリを完全解放
            3. 🎬 **Phase 2**: Wav2Lip でクローン音声と動画を同期し、口パク動画を生成
            4. 🧹 **Final Cleanup**: Phase 2完了後、最終メモリ解放

            ## 📋 使用方法
            1. **動画ファイル**: 口パクさせたい人物の動画をアップロード
            2. **リファレンス音声**: クローンしたい声の音声ファイルをアップロード（3秒以上推奨）
            3. **セリフテキスト**: 生成したいセリフを日本語で入力（5-500文字）
            4. **設定**: GFPGAN・デバイスを選択
            5. **実行**: 統合処理を開始（Phase 1 → Phase 2の順で自動実行）
            """)

            with gr.Row():
                # 左側: 入力とオプション
                with gr.Column(scale=1):
                    gr.Markdown("## 📁 ファイル入力")

                    video_input = gr.File(
                        label="📹 動画ファイル",
                        file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                        type="filepath"
                    )

                    reference_audio_input = gr.File(
                        label="🎵 リファレンス音声ファイル（クローン元）",
                        file_types=[".mp3", ".wav", ".m4a", ".aac", ".flac"],
                        type="filepath"
                    )

                    gr.Markdown("## 📝 セリフ入力")

                    script_input = gr.Textbox(
                        label="セリフテキスト（日本語）",
                        placeholder="例: こんにちは！今日はいい天気ですね。リップシンク技術により、自然な口パク動画を生成します。",
                        lines=4,
                        max_lines=8,
                        elem_classes=["script-input"]
                    )

                    gr.Markdown("### 📝 セリフ入力のコツ")
                    gr.Markdown("""
                    - **文字数**: 5-500文字（推奨: 50-200文字）
                    - **長音**: 「よー」「いくー」形式を使用
                    - **句読点**: 「。」で自然な間を作成
                    - **品質**: 感情豊かな表現が可能
                    """)

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
                        "🎭🎬 統合処理開始（ボイスクローン + 口パク）",
                        variant="primary",
                        size="lg"
                    )

                # 右側: 出力と結果
                with gr.Column(scale=1):
                    gr.Markdown("## 🎥 処理結果")

                    output_video = gr.Video(
                        label="生成されたボイスクローン口パク動画",
                        elem_classes=["output-video"]
                    )

                    download_btn = gr.DownloadButton(
                        label="💾 動画をダウンロード",
                        variant="secondary",
                        visible=False
                    )

                    gr.Markdown("## 📊 統合処理ログ")

                    log_output = gr.Textbox(
                        label="統合処理ログ・統計情報",
                        lines=18,
                        max_lines=25,
                        elem_classes=["log-output"],
                        interactive=False
                    )

            # イベントハンドラー
            def on_process_click(*args):
                """統合処理ボタンクリック時のハンドラー"""
                result = self.process_integrated_pipeline(*args)
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
                    reference_audio_input,
                    script_input,
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
            gr.HTML("""
            <div style="margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px; text-align: center;">
                <h4>🔧 システム仕様</h4>
                <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                    <div>
                        <strong>🎭 SOVITS Voice Clone</strong><br>
                        GPT-SoVITS v4<br>
                        長文対応・高品質音声
                    </div>
                    <div>
                        <strong>🎬 Wav2Lip Lip Sync</strong><br>
                        RetinaFace + ONNX GPU<br>
                        onnxruntime-gpu 1.15.1
                    </div>
                    <div>
                        <strong>⚡ パフォーマンス</strong><br>
                        統合処理: 30-60秒<br>
                        メモリ分離・自動解放
                    </div>
                </div>
                <p style="margin-top: 15px;">
                    <strong>バージョン:</strong> v1.0 - SOVITS-Wav2Lip統合版<br>
                    <strong>開発者:</strong> Claude Code Assistant<br>
                    <strong>更新日:</strong> 2025-09-13
                </p>
            </div>
            """)

        return interface

    def launch(self, **kwargs):
        """統合Gradioアプリケーションの起動"""
        interface = self.create_interface()

        # デフォルト設定
        launch_kwargs = {
            "server_name": "0.0.0.0",
            "server_port": 7864,  # 統合版専用ポート
            "share": False,
            "debug": True,
            "inbrowser": True,
            "allowed_paths": ["/tmp"]  # 一時ディレクトリ許可
        }
        launch_kwargs.update(kwargs)

        print("🎭🎬 SOVITS-Wav2Lip Integration Frontend Starting...")
        print(f"📁 Project Root: {WAV2LIP_ROOT}")
        print(f"📁 Temp Directory: {self.temp_dir}")
        print(f"🌐 Server: http://localhost:{launch_kwargs['server_port']}")
        print(f"🔧 Integration Available: {INTEGRATION_AVAILABLE}")

        interface.launch(**launch_kwargs)


def main():
    """メイン関数"""
    app = SOVITSWav2LipGradioApp()
    app.launch()


if __name__ == "__main__":
    main()