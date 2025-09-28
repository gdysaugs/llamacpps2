#!/usr/bin/env python3
"""
Minimal Wav2Lip Gradio Frontend - Subprocess Integration Test
"""
import gradio as gr
import sys
from pathlib import Path

# Add parent directory to import wav2lip subprocess
WAV2LIP_ROOT = Path(__file__).parent.parent
sys.path.append(str(WAV2LIP_ROOT))

try:
    from wav2lip_subprocess import Wav2LipSubprocess
    SUBPROCESS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import wav2lip_subprocess: {e}")
    SUBPROCESS_AVAILABLE = False

def process_files(video_file, audio_file, use_gfpgan, device):
    """Minimal processing function"""
    if not SUBPROCESS_AVAILABLE:
        return None, "❌ Subprocess module not available", None

    if not video_file or not audio_file:
        return None, "❌ Please upload both video and audio files", None

    try:
        # Initialize subprocess wrapper
        wav2lip = Wav2LipSubprocess()

        # Run inference with temp directory output path (Gradio 5.x compatible)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir='/tmp') as tmp_file:
            output_path = tmp_file.name

        result = wav2lip.run_inference(
            video_file,
            audio_file,
            output_path=output_path,
            use_gfpgan=use_gfpgan,
            device=device
        )

        if result["success"]:
            # Ensure we return absolute path and file exists
            abs_output_path = Path(result["output_path"]).resolve()
            if abs_output_path.exists():
                log_msg = f"✅ Success! Execution time: {result['execution_time']:.2f}s\n📁 Output: {abs_output_path}"
                return str(abs_output_path), log_msg, result
            else:
                log_msg = f"❌ File generated but not found at: {abs_output_path}"
                return None, log_msg, result
        else:
            error_msg = f"❌ Failed: {result.get('error', 'Unknown error')}"
            return None, error_msg, result

    except Exception as e:
        error_msg = f"❌ Exception: {str(e)}"
        return None, error_msg, None

# Create interface
with gr.Blocks(title="Wav2Lip Minimal Test") as interface:
    gr.Markdown("# Wav2Lip Subprocess Test")

    with gr.Row():
        with gr.Column():
            video_input = gr.File(label="Video", file_types=[".mp4", ".avi", ".mov"])
            audio_input = gr.File(label="Audio", file_types=[".mp3", ".wav"])
            use_gfpgan = gr.Checkbox(label="Use GFPGAN", value=True)
            device = gr.Radio(label="Device", choices=["cuda", "cpu"], value="cuda")
            process_btn = gr.Button("Process", variant="primary")

        with gr.Column():
            output_video = gr.Video(label="Result")
            log_output = gr.Textbox(label="Log", lines=10)

    process_btn.click(
        fn=process_files,
        inputs=[video_input, audio_input, use_gfpgan, device],
        outputs=[output_video, log_output, gr.State()]
    )

if __name__ == "__main__":
    print("🚀 Starting Wav2Lip Minimal Test...")
    print(f"📁 Subprocess available: {SUBPROCESS_AVAILABLE}")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False
    )