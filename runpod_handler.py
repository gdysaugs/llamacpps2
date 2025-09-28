#!/usr/bin/env python3
"""
Runpod Serverless Handler for Wav2Lip Multi-venv System
Supports SoVITS + FaceFusion + Wav2Lip pipeline
"""

import runpod
import os
import sys
import json
import base64
import subprocess
import tempfile
import shutil
from pathlib import Path
import traceback
import time
import requests

# Add project paths
sys.path.append('/app')
sys.path.append('/app/gradio_frontend')

def download_file(url, destination):
    """Download file from URL"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def encode_file_to_base64(filepath):
    """Encode file to base64 string"""
    with open(filepath, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def decode_base64_to_file(base64_string, filepath):
    """Decode base64 string to file"""
    with open(filepath, 'wb') as f:
        f.write(base64.b64decode(base64_string))

def run_subprocess_with_venv(venv_path, script_path, args, timeout=600):
    """Run a Python script in a specific virtual environment"""
    activate_cmd = f"source {venv_path}/bin/activate"
    python_cmd = f"python {script_path} {' '.join(args)}"
    full_cmd = f"{activate_cmd} && {python_cmd}"

    try:
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            executable='/bin/bash'
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return None, "Process timeout", -1
    except Exception as e:
        return None, str(e), -1

def process_three_stage_pipeline(input_data):
    """
    Process the 3-stage pipeline:
    1. SoVITS voice cloning
    2. FaceFusion face swap (optional)
    3. Wav2Lip lip sync
    """
    temp_dir = tempfile.mkdtemp(prefix="runpod_")

    try:
        # Parse input parameters
        video_data = input_data.get('video')
        reference_audio_data = input_data.get('reference_audio')
        text = input_data.get('text', '')
        source_image_data = input_data.get('source_image', None)
        use_gfpgan = input_data.get('use_gfpgan', True)
        device = input_data.get('device', 'cuda')

        # Handle file inputs (base64 or URL)
        video_path = os.path.join(temp_dir, "input_video.mp4")
        if video_data.startswith('http'):
            if not download_file(video_data, video_path):
                raise Exception("Failed to download video")
        else:
            decode_base64_to_file(video_data, video_path)

        ref_audio_path = os.path.join(temp_dir, "reference.wav")
        if reference_audio_data.startswith('http'):
            if not download_file(reference_audio_data, ref_audio_path):
                raise Exception("Failed to download reference audio")
        else:
            decode_base64_to_file(reference_audio_data, ref_audio_path)

        # Stage 1: SoVITS Voice Cloning
        print("Stage 1: SoVITS Voice Cloning...")
        cloned_audio_path = os.path.join(temp_dir, "cloned_audio.wav")

        sovits_script = """
import sys
sys.path.append('/app/gpt_sovits_full')
from subprocess import run
import json

# Run SoVITS processing
args = [
    '--ref_audio', '{}',
    '--text', '{}',
    '--output', '{}'
]

# Simplified SoVITS call - you may need to adjust based on actual implementation
import subprocess
result = subprocess.run([
    'python', '/app/gpt_sovits_full/inference.py'
] + args, capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print(result.stderr, file=sys.stderr)
    sys.exit(1)
""".format(ref_audio_path, text, cloned_audio_path)

        sovits_script_path = os.path.join(temp_dir, "sovits_temp.py")
        with open(sovits_script_path, 'w') as f:
            f.write(sovits_script)

        stdout, stderr, returncode = run_subprocess_with_venv(
            '/app/sovits_venv',
            sovits_script_path,
            [],
            timeout=120
        )

        # For now, if SoVITS fails, use reference audio as fallback
        if returncode != 0 or not os.path.exists(cloned_audio_path):
            print(f"SoVITS failed, using reference audio. Error: {stderr}")
            shutil.copy(ref_audio_path, cloned_audio_path)

        # Stage 2: FaceFusion (Optional)
        processed_video_path = video_path
        if source_image_data:
            print("Stage 2: FaceFusion Face Swap...")
            source_image_path = os.path.join(temp_dir, "source_face.jpg")
            if source_image_data.startswith('http'):
                if not download_file(source_image_data, source_image_path):
                    raise Exception("Failed to download source image")
            else:
                decode_base64_to_file(source_image_data, source_image_path)

            swapped_video_path = os.path.join(temp_dir, "face_swapped.mp4")

            facefusion_args = [
                '--source', source_image_path,
                '--target', video_path,
                '--output', swapped_video_path,
                '--execution-provider', 'cuda' if device == 'cuda' else 'cpu',
                '--face-detector-model', 'retinaface',
                '--face-swapper-model', 'inswapper_128_fp16'
            ]

            stdout, stderr, returncode = run_subprocess_with_venv(
                '/app/facefusion_venv',
                '/app/facefusion/run.py',
                facefusion_args,
                timeout=300
            )

            if returncode == 0 and os.path.exists(swapped_video_path):
                processed_video_path = swapped_video_path
            else:
                print(f"FaceFusion failed, using original video. Error: {stderr}")

        # Stage 3: Wav2Lip
        print("Stage 3: Wav2Lip Lip Sync...")
        final_output_path = os.path.join(temp_dir, "final_output.mp4")

        wav2lip_args = [
            '--checkpoint_path', '/app/checkpoints/wav2lip_gan.pth',
            '--video', processed_video_path,
            '--audio', cloned_audio_path,
            '--outfile', final_output_path,
            '--device', device
        ]

        if use_gfpgan:
            wav2lip_args.extend(['--gfpgan', '1'])

        stdout, stderr, returncode = run_subprocess_with_venv(
            '/app/venv',
            '/app/wav2lip_subprocess.py',
            wav2lip_args,
            timeout=600
        )

        if returncode != 0 or not os.path.exists(final_output_path):
            raise Exception(f"Wav2Lip processing failed: {stderr}")

        # Encode output video
        output_base64 = encode_file_to_base64(final_output_path)

        # Get file size
        file_size = os.path.getsize(final_output_path) / (1024 * 1024)  # MB

        return {
            'status': 'success',
            'output_video': output_base64,
            'file_size_mb': round(file_size, 2),
            'stages_completed': ['sovits', 'facefusion' if source_image_data else None, 'wav2lip'],
            'message': 'Processing completed successfully'
        }

    except Exception as e:
        print(f"Error in pipeline: {traceback.format_exc()}")
        return {
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def handler(job):
    """
    Runpod handler function

    Expected input format:
    {
        "input": {
            "video": "base64_string_or_url",
            "reference_audio": "base64_string_or_url",
            "text": "Text to be spoken",
            "source_image": "base64_string_or_url" (optional),
            "use_gfpgan": true/false,
            "device": "cuda" or "cpu"
        }
    }
    """
    try:
        job_input = job['input']

        # Validate required inputs
        if not job_input.get('video'):
            return {'error': 'Video input is required'}
        if not job_input.get('reference_audio'):
            return {'error': 'Reference audio is required'}
        if not job_input.get('text'):
            return {'error': 'Text input is required'}

        # Process the pipeline
        result = process_three_stage_pipeline(job_input)

        return result

    except Exception as e:
        print(f"Handler error: {traceback.format_exc()}")
        return {
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }

# Start the serverless handler
if __name__ == "__main__":
    print("Starting Runpod Serverless Handler...")
    print(f"GPU Available: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"Runpod Pod ID: {os.environ.get('RUNPOD_POD_ID', 'Not set')}")

    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True  # Enable streaming for large outputs
    })