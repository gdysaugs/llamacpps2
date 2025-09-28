#!/usr/bin/env python3
import modal
import os
import sys

app = modal.App("wav2lip-upload-v2")
volume = modal.Volume.from_name("wav2lip-models-2025", create_if_missing=True)

# イメージにファイルを追加
image = modal.Image.debian_slim().pip_install("modal")

if len(sys.argv) > 1:
    file_to_upload = sys.argv[1]
    filename = os.path.basename(file_to_upload)

    # ファイルをイメージに追加
    image = image.add_local_file(file_to_upload, f"/tmp/{filename}")

@app.function(
    image=image,
    volumes={"/storage": volume},
    timeout=7200,  # 2時間
    memory=8192,   # 8GB
)
def upload_file(filename: str):
    """ファイルをストレージにアップロード"""
    local_path = f"/tmp/{filename}"
    storage_path = f"/storage/{filename}"

    print(f"📁 Uploading {filename}")

    if not os.path.exists(local_path):
        print(f"❌ File not found: {local_path}")
        return False

    file_size = os.path.getsize(local_path)
    print(f"📊 File size: {file_size / (1024**3):.2f} GB")

    try:
        # ファイルをコピー
        import shutil
        shutil.copy2(local_path, storage_path)

        # サイズ確認
        uploaded_size = os.path.getsize(storage_path)
        if uploaded_size == file_size:
            print(f"✅ Upload successful! Size: {uploaded_size} bytes")
            volume.commit()
            return True
        else:
            print(f"❌ Size mismatch! Original: {file_size}, Uploaded: {uploaded_size}")
            return False

    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 modal_upload_v2.py <file_path>")
        sys.exit(1)

    filename = os.path.basename(sys.argv[1])

    with app.run():
        result = upload_file.remote(filename)
        if result:
            print(f"🎉 {filename} uploaded successfully to Modal storage!")
        else:
            print(f"❌ {filename} upload failed!")
            sys.exit(1)