#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS サブプロセステスト
gpt_sovits_simple_cli.py をサブプロセスで呼び出すテスト
"""

import subprocess
import sys
from pathlib import Path
import time

def test_sovits_subprocess():
    """サブプロセスでGPT-SoVITSを実行"""

    # プロジェクトルート
    project_root = Path(__file__).parent

    # 仮想環境のPythonパス
    venv_python = project_root / "gpt_sovits_env/bin/python"

    # CLIスクリプトパス
    cli_script = project_root / "gpt_sovits_simple_cli.py"

    # テストパラメータ
    ref_audio = "models/gpt_sovits/baka_new_reference.wav"
    target_text = "サブプロセステストです。これは正常に動作しています。"
    output_file = "subprocess_test.wav"

    print("=" * 60)
    print("GPT-SoVITS サブプロセステスト開始")
    print("=" * 60)
    print(f"📂 作業ディレクトリ: {project_root}")
    print(f"🐍 Python: {venv_python}")
    print(f"📄 スクリプト: {cli_script}")
    print(f"🎤 参照音声: {ref_audio}")
    print(f"📝 生成テキスト: {target_text}")
    print(f"💾 出力ファイル: output/{output_file}")
    print("=" * 60)

    # コマンド構築
    cmd = [
        str(venv_python),
        str(cli_script),
        ref_audio,
        target_text,
        output_file
    ]

    print("🚀 サブプロセス実行中...")
    print(f"コマンド: {' '.join(cmd)}")
    print("-" * 60)

    # 実行時間計測開始
    start_time = time.time()

    try:
        # サブプロセス実行
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
            check=True
        )

        # 実行時間計測終了
        elapsed_time = time.time() - start_time

        print("✅ サブプロセス実行成功！")
        print(f"⏱ 実行時間: {elapsed_time:.2f}秒")
        print("-" * 60)

        # 標準出力を表示
        if result.stdout:
            print("【標準出力】")
            print(result.stdout)

        # 標準エラー出力を表示（警告等）
        if result.stderr:
            print("【標準エラー出力（警告等）】")
            print(result.stderr)

        # 出力ファイル確認
        output_path = project_root / "output" / output_file
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print("-" * 60)
            print(f"✅ 出力ファイル確認: {output_path}")
            print(f"📊 ファイルサイズ: {file_size:.2f}MB")
        else:
            print(f"⚠️ 出力ファイルが見つかりません: {output_path}")

        print("=" * 60)
        print("🎉 テスト完了！")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ サブプロセス実行エラー: {e}")
        print(f"リターンコード: {e.returncode}")
        if e.stdout:
            print("【標準出力】")
            print(e.stdout)
        if e.stderr:
            print("【標準エラー出力】")
            print(e.stderr)
        return False

    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_sovits_subprocess()
    sys.exit(0 if success else 1)