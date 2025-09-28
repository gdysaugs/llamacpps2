#!/usr/bin/env python3
"""
Modal公式推奨のキャッシュを使ったモデルアップロード
"""
import modal
import os
import sys
from pathlib import Path

app = modal.App("wav2lip-cache-upload")

# Volumeを永続ストレージとして使用
volume = modal.Volume.from_name("wav2lip-models-2025", create_if_missing=True)

def download_models():
    """ローカルファイルシステムからモデルをコピー"""
    import shutil
    import subprocess

    models_dir = "/models"
    os.makedirs(models_dir, exist_ok=True)

    # rsyncでファイルを効率的にコピー（利用可能な場合）
    local_models = "/home/adama/wav2lip-project/models"

    print(f"📁 Copying models from {local_models} to {models_dir}")

    try:
        # rsyncが利用可能か確認
        subprocess.run(["which", "rsync"], check=True, capture_output=True)
        # rsyncでコピー
        result = subprocess.run([
            "rsync", "-av", "--progress",
            f"{local_models}/", f"{models_dir}/"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ rsync copy successful")
        else:
            print(f"❌ rsync failed: {result.stderr}")
            raise Exception("rsync failed")

    except (subprocess.CalledProcessError, FileNotFoundError):
        # rsyncが使えない場合は通常のcopy
        print("📋 Using standard copy (rsync not available)")
        shutil.copytree(local_models, models_dir, dirs_exist_ok=True)
        print("✅ Standard copy successful")

# イメージにモデルダウンローダーを追加
image = (
    modal.Image.debian_slim()
    .apt_install("rsync")
    .run_function(download_models, secrets=[])
)

@app.function(
    image=image,
    volumes={"/storage": volume},
    timeout=7200,  # 2時間
    memory=8192,   # 8GB
)
def cache_models_to_storage():
    """キャッシュされたモデルをストレージに移動"""
    import shutil

    models_dir = "/models"
    storage_dir = "/storage"

    print(f"📦 Moving cached models to persistent storage")

    # モデルディレクトリの内容を確認
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return False

    files_copied = 0

    # 全ファイルをストレージにコピー
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            source_path = os.path.join(root, file)
            rel_path = os.path.relpath(source_path, models_dir)
            dest_path = os.path.join(storage_dir, rel_path)

            # ディレクトリ作成
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # ファイルコピー
            print(f"📁 Copying: {rel_path}")
            shutil.copy2(source_path, dest_path)
            files_copied += 1

    print(f"✅ Copied {files_copied} files to storage")

    # ストレージをコミット
    volume.commit()
    print("💾 Storage committed!")

    return True

if __name__ == "__main__":
    print("🚀 Starting model caching and upload...")

    with app.run():
        result = cache_models_to_storage.remote()
        if result:
            print("🎉 All models cached and uploaded successfully!")
        else:
            print("❌ Model upload failed!")
            sys.exit(1)