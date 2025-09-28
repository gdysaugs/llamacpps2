#!/usr/bin/env python3
"""
Wav2Lipサブプロセス版の簡単なテストスクリプト
"""
import os
import time
from pathlib import Path
from wav2lip_subprocess import Wav2LipSubprocess

def test_subprocess():
    """サブプロセス版のテスト実行"""
    print("🧪 Wav2Lip Subprocess Test")
    print("=" * 60)

    # テスト用ファイルパスを設定（実際のファイル名を使用）
    input_dir = Path("input")
    video_path = input_dir / "target_video_3s.mp4"
    audio_path = input_dir / "train_announcement.mp3"
    output_path = "output/test_subprocess_result.mp4"

    print(f"📹 Video: {video_path}")
    print(f"🎵 Audio: {audio_path}")
    print(f"💾 Output: {output_path}")
    print("=" * 60)

    # ファイル存在確認
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        print("📝 Please place a test video file in the input directory")
        return False

    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        print("📝 Please place a test audio file in the input directory")
        return False

    # サブプロセスラッパーを初期化
    wav2lip_subprocess = Wav2LipSubprocess()

    # テスト実行（高速設定）
    print("🚀 Running test with high-speed settings (no GFPGAN)")
    result = wav2lip_subprocess.run_inference(
        video_path=video_path,
        audio_path=audio_path,
        output_path=output_path,
        no_gfpgan=True,  # 高速化のためGFPGAN無効
        device="cuda"
    )

    print("=" * 60)
    print("📊 Test Results:")
    print(f"✅ Success: {result['success']}")
    if result["success"]:
        print(f"⏱️ Execution time: {result['execution_time']:.2f} seconds")
        print(f"📄 Output file: {result['output_path']}")
        # 出力ファイルの存在確認
        if Path(result['output_path']).exists():
            print("✅ Output file created successfully!")
        else:
            print("❌ Output file not found!")
    else:
        print(f"❌ Error: {result['error']}")

    return result["success"]

def test_with_gfpgan():
    """GFPGAN有効でのテスト"""
    print("\n🧪 Wav2Lip Subprocess Test (with GFPGAN)")
    print("=" * 60)

    input_dir = Path("input")
    video_path = input_dir / "target_video_3s.mp4"
    audio_path = input_dir / "train_announcement.mp3"
    output_path = "output/test_subprocess_gfpgan_result.mp4"

    if not video_path.exists() or not audio_path.exists():
        print("❌ Test files not found, skipping GFPGAN test")
        return False

    wav2lip_subprocess = Wav2LipSubprocess()

    print("🚀 Running test with GFPGAN enhancement")
    result = wav2lip_subprocess.run_inference(
        video_path=video_path,
        audio_path=audio_path,
        output_path=output_path,
        gfpgan_blend=0.3,  # 30%ブレンド
        device="cuda"
    )

    print("=" * 60)
    print("📊 GFPGAN Test Results:")
    print(f"✅ Success: {result['success']}")
    if result["success"]:
        print(f"⏱️ Execution time: {result['execution_time']:.2f} seconds")
        print(f"📄 Output file: {result['output_path']}")
    else:
        print(f"❌ Error: {result['error']}")

    return result["success"]

def main():
    """メイン関数"""
    start_time = time.time()

    print("🎬 Wav2Lip Subprocess Testing Suite")
    print("=" * 80)

    # 高速テスト
    test1_success = test_subprocess()

    # GFPGAN付きテスト（時間がかかるので条件付き）
    if test1_success:
        response = input("\n🤔 Run GFPGAN test? (slower but higher quality) [y/N]: ")
        if response.lower() in ['y', 'yes']:
            test_with_gfpgan()

    total_time = time.time() - start_time

    print("=" * 80)
    print("🏁 Testing completed!")
    print(f"⏱️ Total test time: {total_time:.2f} seconds")
    print("=" * 80)

if __name__ == "__main__":
    main()