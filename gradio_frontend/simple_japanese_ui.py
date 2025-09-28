#!/usr/bin/env python3
"""
シンプル日本語UI
音声合成・口パク動画生成・AI会話統合インターフェース
"""
import gradio as gr
import sys
from pathlib import Path

# プロジェクトルートを追加
WAV2LIP_ROOT = Path(__file__).parent.parent
sys.path.append(str(WAV2LIP_ROOT))

try:
    from gradio_frontend.sovits_wav2lip_integration import SOVITSWav2LipIntegration
    from gradio_frontend.llamacpp_integration import LlamaCPPIntegration
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import integration modules: {e}")
    INTEGRATION_AVAILABLE = False

def create_simple_interface():
    """シンプルなGradioインターフェース作成"""

    # 統合システムの初期化
    sovits_integration = SOVITSWav2LipIntegration() if INTEGRATION_AVAILABLE else None
    llama_integration = LlamaCPPIntegration() if INTEGRATION_AVAILABLE else None

    # シンプルなCSS
    css = """
    .japanese-ui {
        font-family: 'Hiragino Sans', 'Meiryo', sans-serif;
    }
    .generate-btn {
        background: #ff6b35 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
    }
    """

    def process_video(video_file, audio_file, script_text, use_ai, additional_prompt, max_tokens):
        """統合処理関数"""
        if not INTEGRATION_AVAILABLE:
            return None, "❌ 統合機能が利用できません"

        try:
            # 入力検証
            if not video_file or not audio_file or not script_text:
                return None, "❌ すべての必須項目を入力してください"

            # AI会話モード
            if use_ai and llama_integration and llama_integration.is_available():
                llama_result = llama_integration.generate_response(
                    user_input=script_text,
                    additional_prompt=additional_prompt,
                    max_tokens=max_tokens
                )

                if llama_result["success"]:
                    final_text = llama_result["response"]
                    log_msg = f"✅ AI応答: {final_text}\n"
                else:
                    return None, f"❌ AI生成エラー: {llama_result['message']}"
            else:
                final_text = script_text
                log_msg = f"✅ 入力テキスト: {final_text}\n"

            # SOVITSとWav2Lip処理
            result = sovits_integration.process_integrated_pipeline(
                video_file=video_file,
                reference_audio_file=audio_file,
                script_text=final_text,
                use_gfpgan=True,
                device="cuda"
            )

            if result[0]:  # 成功
                return result[0], log_msg + "✅ 動画生成完了"
            else:
                return None, log_msg + f"❌ エラー: {result[1]}"

        except Exception as e:
            return None, f"❌ 処理エラー: {str(e)}"

    # インターフェース作成
    with gr.Blocks(css=css, title="AI音声合成・口パク動画生成") as interface:

        gr.Markdown("# 🎭 AI音声合成・口パク動画生成スタジオ")
        gr.Markdown("音声クローン + 口パク生成 + AI会話")

        with gr.Row():
            # 左側: 入力設定
            with gr.Column(scale=1):
                gr.Markdown("## 📝 入力設定")

                script_input = gr.Textbox(
                    label="セリフテキスト",
                    placeholder="AI会話OFF: 音声に変換するテキスト\nAI会話ON: AIへの質問・プロンプト",
                    lines=4
                )

                additional_prompt = gr.Textbox(
                    label="キャラクター特徴",
                    placeholder="明るい、元気、関西弁、猫好き...",
                    lines=2
                )

                video_input = gr.File(
                    label="動画ファイル",
                    file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"]
                )

                reference_audio_input = gr.File(
                    label="リファレンス音声",
                    file_types=[".mp3", ".wav", ".m4a", ".aac", ".flac"]
                )

                gr.Markdown("## ⚙️ AI設定")

                use_ai_conversation = gr.Checkbox(
                    label="AI会話モード",
                    value=False
                )

                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=200,
                    step=10,
                    label="応答文字数"
                )

                generate_button = gr.Button(
                    "🎬 生成開始",
                    elem_classes=["generate-btn"],
                    size="lg"
                )

            # 右側: 出力
            with gr.Column(scale=1):
                gr.Markdown("## 🎥 生成結果")

                output_video = gr.Video(
                    label="生成された動画",
                    height=400
                )

                log_output = gr.Textbox(
                    label="処理ログ",
                    lines=10,
                    interactive=False
                )

        # イベント処理
        generate_button.click(
            fn=process_video,
            inputs=[
                video_input,
                reference_audio_input,
                script_input,
                use_ai_conversation,
                additional_prompt,
                max_tokens
            ],
            outputs=[output_video, log_output]
        )

    return interface

def main():
    """メイン実行"""
    print("🎭 Simple Japanese UI for AI Voice Clone & Lip Sync")
    print("=" * 50)

    try:
        interface = create_simple_interface()

        print("✅ システム初期化完了")
        print("🔧 統合機能利用可能" if INTEGRATION_AVAILABLE else "❌ 統合機能利用不可")

        interface.launch(
            server_name="0.0.0.0",
            server_port=7867,
            share=True,
            show_error=True
        )

    except Exception as e:
        print(f"❌ アプリケーション起動エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()