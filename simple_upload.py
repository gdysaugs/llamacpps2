#!/usr/bin/env python3
import modal
import os
import sys

app = modal.App("wav2lip-simple-upload")
volume = modal.Volume.from_name("wav2lip-models-2025", create_if_missing=True)

@app.function(
    volumes={"/storage": volume},
    timeout=7200,  # 2時間
    memory=8192,   # 8GB
)
def upload_single_file(filename: str):
    """単一ファイルをアップロード"""
    local_path = f"/tmp/{filename}"
    storage_path = f"/storage/{filename}"

    print(f"📁 Uploading {filename}")
    print(f"📊 File size: {os.path.getsize(local_path) / (1024**3):.2f} GB")

    # シンプルコピー
    import shutil
    shutil.copy2(local_path, storage_path)

    # サイズ確認
    if os.path.getsize(storage_path) == os.path.getsize(local_path):
        print("✅ Upload successful!")
        volume.commit()
        return True
    else:
        print("❌ Upload failed - size mismatch")
        return False

# ファイルマウント
if len(sys.argv) > 1:
    file_to_upload = sys.argv[1]
    filename = os.path.basename(file_to_upload)

    # ローカルファイルをマウント
    mount = modal.mount.Mount.from_local_file(file_to_upload, remote_path=f"/tmp/{filename}")
    upload_single_file = upload_single_file.with_mounts([mount])

    if __name__ == "__main__":
        with app.run():
            result = upload_single_file.remote(filename)
            if result:
                print(f"🎉 {filename} uploaded successfully!")
            else:
                print(f"❌ {filename} upload failed!")
else:
    print("Usage: python3 simple_upload.py <file_path>")