#!/usr/bin/env python3
"""
SoVITS サブプロセス版 - 成功したコードをそのまま呼び出し
"""

import subprocess
import sys
from pathlib import Path

def call_sovits_subprocess(text, output_filename):
    """成功したdirect_inference_portable.pyをサブプロセスで呼び出し"""
    base_path = Path(__file__).parent
    python_path = base_path / "gpt_sovits_env" / "bin" / "python"
    script_path = base_path / "direct_inference_separate.py"

    cmd = [
        str(python_path),
        str(script_path),
        text,
        output_filename
    ]

    print(f"サブプロセス実行: {text[:50]}... → {output_filename}")

    result = subprocess.run(
        cmd,
        cwd=str(base_path),
        text=True,
        capture_output=True
    )

    # デバッグ出力
    if result.stdout:
        print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode == 0:
        output_path = base_path / "output" / output_filename
        if output_path.exists():
            return str(output_path)

    return None

def main():
    """メイン関数"""
    if len(sys.argv) < 3:
        print("GPT-SoVITS サブプロセス版（成功コードそのまま使用）")
        print("=" * 50)
        print("使用法: python sovits_subprocess_simple.py <text> <output_filename>")

        # テスト文章（引数がない場合のデフォルト）
        test_text = """ああん！やめてええ！いやだあああ！！！！いやあああ！ゆ、許してください！お願いします！いやあああ！"""
        output_filename = "subprocess_test.wav"

        print(f"テスト開始: {len(test_text)}文字")
    else:
        # コマンドライン引数から取得
        test_text = sys.argv[1]
        output_filename = sys.argv[2]

        print("GPT-SoVITS サブプロセス版（成功コードそのまま使用）")
        print("=" * 50)
        print(f"テキスト: {test_text[:50]}...({len(test_text)}文字)")
        print(f"出力ファイル: {output_filename}")

    result = call_sovits_subprocess(test_text, output_filename)

    if result:
        print(f"🎉 サブプロセス版成功! 出力: {result}")
    else:
        print("❌ サブプロセス版失敗")

if __name__ == "__main__":
    main()