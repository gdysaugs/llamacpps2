#!/usr/bin/env python3
"""
FaceFusion CLI Test Script
動画の顔を画像の顔に入れ替えるテストスクリプト
"""

import os
import sys
import subprocess
from pathlib import Path

# プロジェクトルートディレクトリ
PROJECT_ROOT = Path(__file__).parent
FACEFUSION_DIR = PROJECT_ROOT / "facefusion"

# ファイルパス
SOURCE_IMAGE = PROJECT_ROOT / "input" / "source_face.jpg"
TARGET_VIDEO = PROJECT_ROOT / "input" / "target_video_3s.mp4"
OUTPUT_VIDEO = PROJECT_ROOT / "output" / "facefusion" / "result.mp4"

def check_files():
    """入力ファイルの存在確認"""
    print("=== ファイル確認 ===")
    
    if not SOURCE_IMAGE.exists():
        print(f"❌ ソース画像が見つかりません: {SOURCE_IMAGE}")
        return False
    else:
        print(f"✅ ソース画像: {SOURCE_IMAGE}")
    
    if not TARGET_VIDEO.exists():
        print(f"❌ ターゲット動画が見つかりません: {TARGET_VIDEO}")
        return False
    else:
        print(f"✅ ターゲット動画: {TARGET_VIDEO}")
    
    # 出力ディレクトリ作成
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    print(f"✅ 出力ディレクトリ: {OUTPUT_VIDEO.parent}")
    
    return True

def run_facefusion():
    """FaceFusionを実行"""
    print("\n=== FaceFusion実行 ===")
    
    # FaceFusionのPythonパスを設定
    sys.path.insert(0, str(FACEFUSION_DIR))
    os.chdir(FACEFUSION_DIR)
    
    # FaceFusionコマンド構築 (事前ダウンロード済みモデル使用)
    cmd = [
        sys.executable,
        "facefusion.py",
        "headless-run",
        "--source-paths", str(SOURCE_IMAGE),
        "--target-path", str(TARGET_VIDEO),
        "--output-path", str(OUTPUT_VIDEO),
        "--execution-providers", "cuda",  # GPU使用
        "--processors", "face_swapper",
        "--face-swapper-model", "inswapper_128",
        "--output-video-encoder", "libx264",
        "--output-video-quality", "18",  # 高品質維持
        "--system-memory-limit", "8",  # メモリ制限
        "--execution-thread-count", "4",  # スレッド数
        "--video-memory-strategy", "moderate",
        "--log-level", "info"
    ]
    
    print("実行コマンド:")
    print(" ".join(cmd))
    
    try:
        # FaceFusion実行
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        print("\n--- 標準出力 ---")
        print(result.stdout)
        
        if result.stderr:
            print("\n--- エラー出力 ---")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n✅ 処理成功！出力: {OUTPUT_VIDEO}")
            return True
        else:
            print(f"\n❌ 処理失敗 (終了コード: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        return False

def main():
    """メイン処理"""
    print("=== FaceFusion CLI テスト ===")
    print(f"Python: {sys.executable}")
    print(f"作業ディレクトリ: {os.getcwd()}")
    
    # ファイル確認
    if not check_files():
        sys.exit(1)
    
    # FaceFusion実行
    if run_facefusion():
        print("\n✨ テスト完了!")
        print(f"結果を確認: {OUTPUT_VIDEO}")
    else:
        print("\n❌ テスト失敗")
        sys.exit(1)

if __name__ == "__main__":
    main()