#!/usr/bin/env python3
import modal
import os
import sys

app = modal.App("wav2lip-bucket-upload")

# CloudBucketMountでS3バケットをマウント
bucket_mount = modal.CloudBucketMount(
    "s3://wav2lip-models-cache",  # S3バケット名
    secret=modal.Secret.from_name("aws-credentials"),  # AWS認証情報
)

# Volume for persistent storage
volume = modal.Volume.from_name("wav2lip-models-2025", create_if_missing=True)

@app.function(
    cloud_bucket_mounts={"/bucket": bucket_mount},
    volumes={"/storage": volume},
    timeout=7200,  # 2時間
    memory=8192,   # 8GB
)
def upload_via_bucket(filename: str, local_file_content: bytes):
    """CloudBucketMount経由でファイルをアップロード"""
    bucket_path = f"/bucket/{filename}"
    storage_path = f"/storage/{filename}"

    print(f"📁 Uploading {filename} via CloudBucket")
    print(f"📊 File size: {len(local_file_content) / (1024**3):.2f} GB")

    try:
        # まずバケットに書き込み
        with open(bucket_path, 'wb') as f:
            f.write(local_file_content)
        print(f"✅ Written to bucket: {bucket_path}")

        # バケットからストレージにコピー
        import shutil
        shutil.copy2(bucket_path, storage_path)
        print(f"✅ Copied to storage: {storage_path}")

        # サイズ確認
        uploaded_size = os.path.getsize(storage_path)
        if uploaded_size == len(local_file_content):
            print(f"✅ Upload successful! Size: {uploaded_size} bytes")
            volume.commit()
            return True
        else:
            print(f"❌ Size mismatch! Original: {len(local_file_content)}, Uploaded: {uploaded_size}")
            return False

    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 modal_bucket_upload.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    filename = os.path.basename(file_path)

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    # ファイルを読み込み
    print(f"📖 Reading file: {file_path}")
    with open(file_path, 'rb') as f:
        file_content = f.read()

    print(f"📊 File size: {len(file_content) / (1024**3):.2f} GB")

    with app.run():
        result = upload_via_bucket.remote(filename, file_content)
        if result:
            print(f"🎉 {filename} uploaded successfully!")
        else:
            print(f"❌ {filename} upload failed!")
            sys.exit(1)