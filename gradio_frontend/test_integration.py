#!/usr/bin/env python3
"""
Wav2Lip Gradio Integration Test
フロントエンドとバックエンドの統合テスト
"""

import subprocess
import time
from pathlib import Path

# Test configuration
WAV2LIP_ROOT = Path(__file__).parent.parent
MAIN_PYTHON = WAV2LIP_ROOT / "venv" / "bin" / "python"
SUBPROCESS_SCRIPT = WAV2LIP_ROOT / "wav2lip_subprocess.py"

# Test files
TEST_VIDEO = WAV2LIP_ROOT / "input" / "target_video_3s.mp4"
TEST_AUDIO = WAV2LIP_ROOT / "input" / "train_announcement.mp3"
OUTPUT_PATH = "/tmp/wav2lip_gradio/integration_test_result.mp4"

def test_subprocess_integration():
    """Test the subprocess integration directly"""
    print("🧪 Testing Wav2Lip subprocess integration...")

    # Check if test files exist
    if not TEST_VIDEO.exists():
        print(f"❌ Test video not found: {TEST_VIDEO}")
        return False

    if not TEST_AUDIO.exists():
        print(f"❌ Test audio not found: {TEST_AUDIO}")
        return False

    # Build command
    cmd = [
        str(MAIN_PYTHON),
        str(SUBPROCESS_SCRIPT),
        str(TEST_VIDEO),
        str(TEST_AUDIO),
        "-o", OUTPUT_PATH,
        "--no-gfpgan",  # Fast test
        "--device", "cuda"
    ]

    print(f"📋 Command: {' '.join(cmd)}")

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(WAV2LIP_ROOT)
        )
        end_time = time.time()

        print(f"✅ Integration test successful!")
        print(f"⏱️ Execution time: {end_time - start_time:.2f} seconds")
        print(f"📄 Output: {OUTPUT_PATH}")

        # Check if output file exists
        if Path(OUTPUT_PATH).exists():
            print(f"✅ Output file created successfully")
            file_size = Path(OUTPUT_PATH).stat().st_size / (1024 * 1024)
            print(f"📦 File size: {file_size:.2f} MB")
            return True
        else:
            print(f"❌ Output file not found")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Integration test failed!")
        print(f"💥 Return code: {e.returncode}")
        print(f"📝 stdout: {e.stdout}")
        print(f"📝 stderr: {e.stderr}")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_environment_paths():
    """Test environment and path configuration"""
    print("🔍 Testing environment paths...")

    # Check main Python venv
    if MAIN_PYTHON.exists():
        print(f"✅ Main Python venv found: {MAIN_PYTHON}")
    else:
        print(f"❌ Main Python venv not found: {MAIN_PYTHON}")
        return False

    # Check subprocess script
    if SUBPROCESS_SCRIPT.exists():
        print(f"✅ Subprocess script found: {SUBPROCESS_SCRIPT}")
    else:
        print(f"❌ Subprocess script not found: {SUBPROCESS_SCRIPT}")
        return False

    # Test main Python import
    try:
        import_test_cmd = [
            str(MAIN_PYTHON), "-c",
            "import cv2, numpy, onnxruntime; print('✅ All dependencies available')"
        ]
        result = subprocess.run(import_test_cmd, capture_output=True, text=True, check=True)
        print(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Dependency test failed: {e.stderr}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 Wav2Lip Gradio Integration Test Suite")
    print("=" * 60)

    # Test 1: Environment paths
    if not test_environment_paths():
        print("❌ Environment test failed")
        return

    print("\n" + "=" * 60)

    # Test 2: Subprocess integration
    if not test_subprocess_integration():
        print("❌ Integration test failed")
        return

    print("\n" + "=" * 60)
    print("✅ All tests passed! Gradio frontend should work correctly.")
    print("🌐 Access the frontend at: http://localhost:7862")
    print("=" * 60)

if __name__ == "__main__":
    main()