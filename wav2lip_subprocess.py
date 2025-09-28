#!/usr/bin/env python3
"""
Wav2Lip Subprocess Wrapper - onnxruntime-gpu 1.15.1 Compatible
Simple interface to call the working wav2lip command via subprocess
"""
import subprocess
import sys
import os
import time
from pathlib import Path

class Wav2LipSubprocess:
    def __init__(self, script_path=None):
        """
        Args:
            script_path: wav2lip_retinaface_gpu.pyのパス（Noneの場合は自動検出）
        """
        if script_path is None:
            # 自動でwav2lip_retinaface_gpu.pyを検出
            current_dir = Path(__file__).parent
            self.script_path = current_dir / "wav2lip_retinaface_gpu.py"
        else:
            self.script_path = Path(script_path)

        if not self.script_path.exists():
            raise FileNotFoundError(f"wav2lip_retinaface_gpu.py not found: {self.script_path}")

    def run_inference(self, video_path, audio_path, output_path="output/result.mp4", use_gfpgan=True, device="cuda"):
        """
        Working wav2lip command をサブプロセスで実行

        Args:
            video_path: 入力動画パス
            audio_path: 入力音声パス
            output_path: 出力動画パス
            use_gfpgan: GFPGAN使用フラグ
            device: デバイス ('cuda' or 'cpu')
        """
        print("=" * 60)
        print("🚀 Starting Wav2Lip subprocess with onnxruntime-gpu 1.15.1")
        print("=" * 60)

        # Get project paths
        current_dir = Path(__file__).parent
        checkpoint_path = current_dir / "models" / "wav2lip" / "wav2lip_gan.onnx"
        venv_activate = current_dir / "venv" / "bin" / "activate"

        # Verify required files
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not venv_activate.exists():
            raise FileNotFoundError(f"Virtual environment not found: {venv_activate}")

        # Build the bash command with venv activation
        cmd_parts = [
            f"source {venv_activate}",
            "&&",
            "python", str(self.script_path),
            "--checkpoint_path", str(checkpoint_path),
            "--face", f'"{video_path}"',
            "--audio", f'"{audio_path}"',
            "--outfile", f'"{output_path}"',
            "--device", device
        ]

        if use_gfpgan:
            cmd_parts.append("--gfpgan")

        # Join into single bash command
        bash_cmd = " ".join(cmd_parts)
        cmd = ["bash", "-c", bash_cmd]

        print(f"📋 Command: {' '.join(cmd)}")
        print("=" * 60)

        # サブプロセス実行
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
                cwd=current_dir  # Set working directory
            )

            end_time = time.time()
            execution_time = end_time - start_time

            success = result.returncode == 0

            print("=" * 60)
            if success:
                print("✅ Subprocess completed successfully!")
            else:
                print("❌ Subprocess failed!")
            print(f"⏱️ Execution time: {execution_time:.2f} seconds")
            print(f"💥 Return code: {result.returncode}")
            print("=" * 60)

            if result.stdout:
                print("📄 Stdout:")
                print(result.stdout)
            if result.stderr:
                print("📄 Stderr:")
                print(result.stderr)

            return {
                "success": success,
                "output_path": output_path,
                "execution_time": execution_time,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

        except subprocess.TimeoutExpired as e:
            return {
                "success": False,
                "error": "Process timed out after 10 minutes",
                "execution_time": 600,
                "stdout": e.stdout if e.stdout else "",
                "stderr": e.stderr if e.stderr else ""
            }
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time

            print("=" * 60)
            print("❌ Subprocess failed with exception!")
            print(f"⏱️ Execution time: {execution_time:.2f} seconds")
            print(f"💥 Error: {str(e)}")
            print("=" * 60)

            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "stdout": "",
                "stderr": str(e)
            }

def main():
    """コマンドライン用のメイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='Wav2Lip Subprocess Wrapper - onnxruntime-gpu 1.15.1')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('audio_path', help='Path to input audio file')
    parser.add_argument('-o', '--output', default='output/result_subprocess.mp4',
                        help='Output video path (default: output/result_subprocess.mp4)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--no-gfpgan', action='store_true',
                        help='Disable GFPGAN enhancement')

    args = parser.parse_args()

    # サブプロセスラッパーを初期化
    wav2lip_subprocess = Wav2LipSubprocess()

    # 推論実行
    result = wav2lip_subprocess.run_inference(
        args.video_path,
        args.audio_path,
        args.output,
        use_gfpgan=not args.no_gfpgan,
        device=args.device
    )

    if not result["success"]:
        print("Process failed!")
        sys.exit(1)
    else:
        print(f"✅ Process completed successfully!")
        print(f"📁 Output saved to: {result['output_path']}")

# Quick test function
def quick_test():
    """Quick test function for development"""
    video_path = "/mnt/c/Users/adama/Videos/画面録画/画面録画 2025-06-29 150438.mp4"
    audio_path = "/mnt/c/Users/adama/Downloads/「間もなく、次の駅に到着いたします。お忘れ物のないようお降りください」.mp3"
    output_path = "output/subprocess_test.mp4"

    wav2lip = Wav2LipSubprocess()
    result = wav2lip.run_inference(video_path, audio_path, output_path)

    if result["success"]:
        print(f"✅ Test completed! Output: {output_path}")
    else:
        print("❌ Test failed!")

    return result

if __name__ == "__main__":
    main()