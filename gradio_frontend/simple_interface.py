#!/usr/bin/env python3
"""
シンプルなgr.Interface版
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

def process_audio_video(video_file, audio_file, script_text, use_ai, additional_prompt, max_tokens):
    """統合処理関数"""
    if not INTEGRATION_AVAILABLE:
        return None

    try:
        # 入力検証
        if not video_file or not audio_file or not script_text:
            return None

        # 統合システムの初期化
        sovits_integration = SOVITSWav2LipIntegration()
        llama_integration = LlamaCPPIntegration()

        # AI会話モード
        if use_ai and llama_integration and llama_integration.is_available():
            llama_result = llama_integration.generate_response(
                user_input=script_text,
                additional_prompt=additional_prompt,
                max_tokens=max_tokens
            )

            if llama_result["success"]:
                final_text = llama_result["response"]
            else:
                return None
        else:
            final_text = script_text

        # SOVITSとWav2Lip処理
        result = sovits_integration.process_integrated_pipeline(
            video_file=video_file,
            reference_audio_file=audio_file,
            script_text=final_text,
            use_gfpgan=True,
            device="cuda"
        )

        if result[0]:  # 成功
            return result[0]
        else:
            return None

    except Exception as e:
        print(f"エラー: {e}")
        return None

# インターフェース作成
interface = gr.Interface(
    fn=process_audio_video,
    inputs=[
        gr.File(label="動画ファイル"),
        gr.File(label="リファレンス音声"),
        gr.Textbox(label="セリフテキスト", lines=4),
        gr.Checkbox(label="AI会話モード"),
        gr.Textbox(label="キャラクター特徴", lines=2),
        gr.Slider(50, 500, 200, label="応答文字数")
    ],
    outputs=gr.Video(label="生成された動画"),
    title="🎭 AI音声合成・口パク動画生成",
    description="音声クローン + 口パク生成 + AI会話"
)

if __name__ == "__main__":
    print("🎭 Simple Interface for AI Voice Clone & Lip Sync")
    print("=" * 50)

    interface.launch(
        server_name="0.0.0.0",
        server_port=7868,
        share=True
    )