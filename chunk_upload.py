#!/usr/bin/env python3
"""
チャンク分割でModal storageにファイルをアップロードするスクリプト
"""

import os
import sys
import time
from pathlib import Path
import modal

# Modal設定
app = modal.App("wav2lip-chunk-uploader")
storage_volume = modal.Volume.from_name("wav2lip-models-2025", create_if_missing=True)

@app.function(
    volumes={"/storage": storage_volume},
    timeout=3600,  # 1時間タイムアウト
    memory=4096,   # 4GB memory
    mounts=[modal.Mount.from_local_file(file_path, remote_path=f"/tmp/{os.path.basename(file_path)}") for file_path in ["/home/adama/wav2lip-project/models/Berghof-NSFW-7B.i1-IQ4_XS.gguf"]]
)
def upload_file_chunks(target_name: str, chunk_size: int = 100 * 1024 * 1024):  # 100MB chunks
    """ファイルをチャンクに分割してアップロード"""
    import shutil

    print(f"📁 Starting chunked upload: {file_path} -> {target_name}")
    print(f"📦 Chunk size: {chunk_size / (1024*1024):.1f} MB")

    storage_path = f"/storage/{target_name}"
    temp_dir = f"/storage/temp_{target_name.replace('/', '_')}"

    try:
        # テンポラリディレクトリ作成
        os.makedirs(temp_dir, exist_ok=True)

        # ファイルサイズ取得
        file_size = os.path.getsize(file_path)
        total_chunks = (file_size + chunk_size - 1) // chunk_size

        print(f"📊 File size: {file_size / (1024*1024*1024):.2f} GB")
        print(f"🔢 Total chunks: {total_chunks}")

        # チャンクに分割してアップロード
        with open(file_path, 'rb') as source_file:
            for chunk_num in range(total_chunks):
                chunk_data = source_file.read(chunk_size)
                if not chunk_data:
                    break

                chunk_path = f"{temp_dir}/chunk_{chunk_num:04d}"
                with open(chunk_path, 'wb') as chunk_file:
                    chunk_file.write(chunk_data)

                print(f"✅ Chunk {chunk_num + 1}/{total_chunks} uploaded ({len(chunk_data)} bytes)")

        # チャンクを結合して元ファイルを復元
        print("🔗 Combining chunks...")
        with open(storage_path, 'wb') as output_file:
            for chunk_num in range(total_chunks):
                chunk_path = f"{temp_dir}/chunk_{chunk_num:04d}"
                with open(chunk_path, 'rb') as chunk_file:
                    output_file.write(chunk_file.read())
                os.remove(chunk_path)  # チャンクファイル削除

        # テンポラリディレクトリ削除
        os.rmdir(temp_dir)

        # ファイルサイズ確認
        uploaded_size = os.path.getsize(storage_path)
        if uploaded_size == file_size:
            print(f"🎉 Upload successful! Size verified: {uploaded_size} bytes")
        else:
            print(f"❌ Size mismatch! Original: {file_size}, Uploaded: {uploaded_size}")
            return False

        # ボリューム変更をコミット
        storage_volume.commit()
        print("💾 Storage volume committed!")
        return True

    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        # クリーンアップ
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except:
            pass
        return False

@app.function(
    volumes={"/storage": storage_volume},
    timeout=7200,  # 2時間タイムアウト
    memory=2048,
)
def upload_directory(local_dir: str, target_dir: str = ""):
    """ディレクトリを再帰的にアップロード"""
    print(f"📁 Uploading directory: {local_dir} -> /{target_dir}")

    uploaded_files = []

    for root, dirs, files in os.walk(local_dir):
        for file in files:
            source_path = os.path.join(root, file)
            relative_path = os.path.relpath(source_path, local_dir)

            if target_dir:
                target_path = f"{target_dir}/{relative_path}"
            else:
                target_path = relative_path

            storage_path = f"/storage/{target_path}"

            # ディレクトリ作成
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)

            # ファイルコピー
            import shutil
            shutil.copy2(source_path, storage_path)
            uploaded_files.append(target_path)
            print(f"✅ {relative_path}")

    storage_volume.commit()
    print(f"🎉 Directory upload complete! {len(uploaded_files)} files uploaded")
    return uploaded_files

def main():
    if len(sys.argv) < 3:
        print("Usage: python chunk_upload.py <file_path> <target_name>")
        print("   or: python chunk_upload.py --dir <directory_path> [target_dir]")
        sys.exit(1)

    if sys.argv[1] == "--dir":
        # ディレクトリアップロード
        local_dir = sys.argv[2]
        target_dir = sys.argv[3] if len(sys.argv) > 3 else ""

        with app.run():
            result = upload_directory.remote(local_dir, target_dir)
            if result:
                print(f"✅ Directory upload completed successfully!")
    else:
        # ファイルアップロード
        file_path = sys.argv[1]
        target_name = sys.argv[2]

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            sys.exit(1)

        with app.run():
            result = upload_file_chunks.remote(file_path, target_name)
            if result:
                print(f"✅ File upload completed successfully!")
            else:
                print(f"❌ File upload failed!")
                sys.exit(1)

if __name__ == "__main__":
    main()