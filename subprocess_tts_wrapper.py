#!/usr/bin/env python3
"""
サブプロセスでポータブルTTSを実行するラッパー
"""
import subprocess
import sys
import json
import os
from pathlib import Path
import time

class SubprocessTTS:
    def __init__(self):
        self.base_dir = Path("/home/adama/wav2lip-project")
        self.venv_python = self.base_dir / "gpt_sovits_env" / "bin" / "python"
        self.portable_script = self.base_dir / "direct_inference_portable.py"
        self.separate_script = self.base_dir / "direct_inference_separate.py"
        
    def generate_with_portable(self, text, output_path, use_separate=False):
        """
        サブプロセスでポータブル版TTSを実行
        
        Args:
            text: 生成するテキスト
            output_path: 出力WAVファイルパス
            use_separate: 長文対応版を使用するか
        """
        script = self.separate_script if use_separate else self.portable_script
        
        # 一時的なPythonスクリプトを作成して実行
        temp_script = self.base_dir / "temp_tts_runner.py"
        
        script_content = f"""
import sys
sys.path.insert(0, '{self.base_dir}')
{'from direct_inference_separate import SeparateTTS as PortableTTS' if use_separate else 'from direct_inference_portable import PortableTTS'}

# TTSインスタンスを作成
tts = PortableTTS()
tts.init_tts()

# 音声生成
text = '''{text}'''
output_path = '{output_path}'

print(f"Generating speech for: {{text[:50]}}...")
{'tts.generate_separate_speech(text, output_path)' if use_separate else 'tts.generate_speech(text, output_path)'}
print(f"Speech generated: {{output_path}}")
"""
        
        # 一時スクリプトを書き込み
        with open(temp_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        try:
            # サブプロセスで実行
            cmd = [str(self.venv_python), str(temp_script)]
            
            print(f"Running TTS in subprocess...")
            print(f"Command: {' '.join(cmd)}")
            
            # サブプロセスを実行
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                cwd=str(self.base_dir)
            )
            
            # リアルタイムで出力を表示
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(f"[SUBPROCESS] {output.strip()}")
            
            # プロセスの終了を待つ
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"Error in subprocess: {stderr}")
                return False
            
            # 出力ファイルの確認
            if Path(output_path).exists():
                file_size = Path(output_path).stat().st_size / 1024
                print(f"✅ Success! File generated: {output_path} ({file_size:.2f} KB)")
                return True
            else:
                print(f"❌ Failed: Output file not created")
                return False
                
        finally:
            # 一時スクリプトを削除
            if temp_script.exists():
                temp_script.unlink()
    
    def run_api_server_subprocess(self):
        """
        APIサーバーをサブプロセスとして起動
        """
        cmd = [
            str(self.venv_python),
            str(self.base_dir / "gpt_sovits_full" / "api_v2.py"),
            "-a", "127.0.0.1",
            "-p", "9880",
            "-c", "GPT_SoVITS/configs/tts_infer.yaml"
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.base_dir / "gpt_sovits_full")
        
        print("Starting API server in subprocess...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            cwd=str(self.base_dir / "gpt_sovits_full"),
            env=env
        )
        
        print(f"API server started with PID: {process.pid}")
        return process

def main():
    """テスト実行"""
    tts = SubprocessTTS()
    
    # テストケース
    test_cases = [
        {
            "text": "こんにちは。サブプロセスでの音声生成テストです。",
            "output": "/home/adama/wav2lip-project/output/subprocess_test1.wav",
            "use_separate": False
        },
        {
            "text": "ああっ、気持ちいい。もっと、もっとして。サブプロセスでも正常に動作しています。",
            "output": "/home/adama/wav2lip-project/output/subprocess_test2.wav",
            "use_separate": False
        },
        {
            "text": "これは長文テストです。サブプロセスで実行することで、メインプロセスをブロックせずに音声生成が可能になります。また、複数のプロセスを並列実行することも可能です。このような長い文章でも問題なく生成できることを確認します。さらに、エラーハンドリングも適切に行われ、プロセスの終了ステータスも確認できます。",
            "output": "/home/adama/wav2lip-project/output/subprocess_test3_long.wav",
            "use_separate": True  # 長文なので分割版を使用
        }
    ]
    
    print("=" * 60)
    print("Subprocess TTS Wrapper Test")
    print("=" * 60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}]")
        print(f"Text: {test['text'][:50]}...")
        print(f"Using: {'Separate (Long text)' if test['use_separate'] else 'Standard'} version")
        
        start_time = time.time()
        success = tts.generate_with_portable(
            test['text'],
            test['output'],
            test['use_separate']
        )
        elapsed = time.time() - start_time
        
        if success:
            print(f"⏱️  Generation time: {elapsed:.2f} seconds")
        else:
            print(f"❌ Test {i} failed")
        
        print("-" * 40)
    
    print("\n✨ All tests completed!")

if __name__ == "__main__":
    main()