#!/usr/bin/env python3
"""
ポータブル版依存関係インストールスクリプト
初回セットアップ時に実行
"""
import subprocess
import sys
import os
from pathlib import Path
import platform

def get_app_root():
    """アプリケーションルートディレクトリを取得"""
    return Path(__file__).parent

def get_python_executable():
    """ポータブルPython実行ファイルを取得"""
    app_root = get_app_root()
    system = platform.system()

    if system == "Windows":
        python_exe = app_root / "python" / "python.exe"
    else:
        python_exe = app_root / "python" / "bin" / "python"

    if python_exe.exists():
        return str(python_exe)
    else:
        return sys.executable

def install_requirements():
    """requirements.txtから依存関係をインストール"""
    app_root = get_app_root()
    python_exe = get_python_executable()
    requirements_file = app_root / "gradio_frontend" / "requirements.txt"

    if not requirements_file.exists():
        print(f"❌ requirements.txtが見つかりません: {requirements_file}")
        return False

    print(f"📦 依存関係をインストール中...")
    print(f"Python: {python_exe}")
    print(f"Requirements: {requirements_file}")

    try:
        # pip install実行
        cmd = [python_exe, "-m", "pip", "install", "-r", str(requirements_file)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        print("✅ 依存関係のインストールが完了しました")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ 依存関係のインストールに失敗しました:")
        print(f"エラーコード: {e.returncode}")
        print(f"エラー出力: {e.stderr}")
        return False

def install_pytorch():
    """PyTorchをCUDA対応でインストール"""
    python_exe = get_python_executable()

    print("🔥 PyTorch (CUDA対応版) をインストール中...")

    try:
        # PyTorch CUDA版をインストール
        cmd = [
            python_exe, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ PyTorch (CUDA対応版) のインストールが完了しました")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorchのインストールに失敗しました:")
        print(f"CPU版をインストールします...")

        try:
            # CPU版をフォールバック
            cmd = [python_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
            subprocess.run(cmd, check=True)
            print("✅ PyTorch (CPU版) のインストールが完了しました")
            return True
        except:
            print("❌ PyTorchのインストールに完全に失敗しました")
            return False

def create_directories():
    """必要なディレクトリを作成"""
    app_root = get_app_root()

    directories = [
        "output",
        "temp",
        "models",
        "models/wav2lip",
        "models/gpt_sovits",
        "models/facefusion"
    ]

    for dir_name in directories:
        dir_path = app_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 ディレクトリ作成: {dir_path}")

    print("✅ ディレクトリ構造の作成が完了しました")

def main():
    """メイン関数"""
    print("=" * 60)
    print("  SoVITS-Wav2Lip-LlamaCPP ポータブル版セットアップ")
    print("=" * 60)
    print()

    app_root = get_app_root()
    print(f"📁 アプリケーションルート: {app_root}")
    print(f"💻 OS: {platform.system()} {platform.release()}")
    print()

    # ディレクトリ作成
    print("1️⃣ 必要なディレクトリを作成中...")
    create_directories()
    print()

    # PyTorchインストール
    print("2️⃣ PyTorchをインストール中...")
    pytorch_success = install_pytorch()
    print()

    # その他の依存関係インストール
    print("3️⃣ その他の依存関係をインストール中...")
    deps_success = install_requirements()
    print()

    # 結果表示
    print("=" * 60)
    print("  セットアップ結果")
    print("=" * 60)
    print(f"PyTorch: {'✅ 成功' if pytorch_success else '❌ 失敗'}")
    print(f"依存関係: {'✅ 成功' if deps_success else '❌ 失敗'}")
    print()

    if pytorch_success and deps_success:
        print("🎉 セットアップが正常に完了しました！")
        print("📝 次の手順:")
        print("   1. モデルファイルをダウンロード")
        print("   2. run_wav2lip.bat (Windows) または run_wav2lip.sh (Linux) を実行")
    else:
        print("⚠️ セットアップ中にエラーが発生しました。")
        print("📝 手動でインストールしてください:")
        print("   pip install torch torchvision torchaudio")
        print("   pip install -r gradio_frontend/requirements.txt")

    input("\n何かキーを押して終了...")

if __name__ == "__main__":
    main()