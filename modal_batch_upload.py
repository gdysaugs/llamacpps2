#!/usr/bin/env python3
"""
Modal Volume batch_upload を使った効率的なアップロード
"""
import modal
import os
import sys
from pathlib import Path

def upload_models():
    """modelsディレクトリを batch_upload でアップロード"""

    # Volumeに接続
    vol = modal.Volume.from_name("wav2lip-models-2025", create_if_missing=True)

    models_dir = "/home/adama/wav2lip-project/models"

    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return False

    print(f"📁 Starting batch upload from {models_dir}")

    try:
        with vol.batch_upload() as batch:
            print("📦 Adding files to batch...")

            # 各ファイルを個別に追加
            for root, dirs, files in os.walk(models_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, models_dir)
                    remote_path = f"/{relative_path}"

                    file_size = os.path.getsize(local_path)
                    print(f"📄 Adding: {relative_path} ({file_size / (1024**2):.1f} MB)")

                    batch.put_file(local_path, remote_path)

            print("🚀 Executing batch upload...")

        print("✅ Batch upload completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Batch upload failed: {str(e)}")
        return False

def upload_large_file(file_path: str):
    """大きなファイルを個別にアップロード"""

    vol = modal.Volume.from_name("wav2lip-models-2025", create_if_missing=True)

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False

    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    print(f"📁 Uploading large file: {filename}")
    print(f"📊 File size: {file_size / (1024**3):.2f} GB")

    try:
        with vol.batch_upload() as batch:
            print("📦 Adding large file to batch...")
            batch.put_file(file_path, f"/{filename}")
            print("🚀 Executing upload...")

        print("✅ Large file upload completed!")
        return True

    except Exception as e:
        print(f"❌ Large file upload failed: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Upload models to Modal storage')
    parser.add_argument('--file', help='Upload specific file')
    parser.add_argument('--all', action='store_true', help='Upload all models directory')

    args = parser.parse_args()

    if args.file:
        # 特定ファイルをアップロード
        success = upload_large_file(args.file)
    elif args.all:
        # 全ディレクトリをアップロード
        success = upload_models()
    else:
        print("Usage:")
        print("  python3 modal_batch_upload.py --file <file_path>")
        print("  python3 modal_batch_upload.py --all")
        sys.exit(1)

    if success:
        print("🎉 Upload operation completed successfully!")
    else:
        print("❌ Upload operation failed!")
        sys.exit(1)