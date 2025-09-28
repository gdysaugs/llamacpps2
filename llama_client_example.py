#!/usr/bin/env python3
"""
Llama Subprocess Client Examples
サブプロセスでLlamaを呼び出す使用例
"""

import subprocess
import json
import sys
import os

class LlamaClient:
    def __init__(self, venv_path="/home/adama/wav2lip-project/llama_venv"):
        self.venv_path = venv_path
        self.python_path = f"{venv_path}/bin/python"

    def single_generation(self, prompt, max_tokens=512, temperature=0.7):
        """単発テキスト生成"""
        cmd = [
            self.python_path,
            "llama_subprocess.py",
            "--mode", "single",
            "--prompt", prompt,
            "--max_tokens", str(max_tokens),
            "--temperature", str(temperature)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="/home/adama/wav2lip-project"
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                raise Exception(f"Error: {result.stderr}")

        except Exception as e:
            raise Exception(f"Subprocess error: {str(e)}")

    def interactive_session(self):
        """インタラクティブセッション"""
        cmd = [
            self.python_path,
            "llama_subprocess.py",
            "--mode", "interactive"
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/home/adama/wav2lip-project"
        )

        return process

    def json_session(self):
        """JSONモードセッション"""
        cmd = [
            self.python_path,
            "llama_subprocess.py",
            "--mode", "json"
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/home/adama/wav2lip-project"
        )

        return process

def example_single():
    """単発生成の例"""
    print("=== 単発生成の例 ===")
    client = LlamaClient()

    try:
        response = client.single_generation(
            "今日は良い天気ですね。",
            max_tokens=100,
            temperature=0.7
        )
        print(f"入力: 今日は良い天気ですね。")
        print(f"応答: {response}")
    except Exception as e:
        print(f"エラー: {e}")

def example_interactive():
    """インタラクティブセッションの例"""
    print("\n=== インタラクティブセッションの例 ===")
    client = LlamaClient()

    try:
        process = client.interactive_session()

        # 複数のプロンプトを送信
        prompts = [
            "こんにちは",
            "今日の天気はどうですか？",
            "EXIT"
        ]

        for prompt in prompts:
            if prompt != "EXIT":
                print(f"送信: {prompt}")

            process.stdin.write(prompt + "\n")
            process.stdin.flush()

            if prompt == "EXIT":
                break

            # 応答を読み取り
            while True:
                line = process.stdout.readline()
                if line.startswith("RESPONSE:"):
                    response = line[9:].strip()
                    print(f"応答: {response}")
                elif line.strip() == "END_RESPONSE":
                    break

        process.wait()

    except Exception as e:
        print(f"エラー: {e}")

def example_json():
    """JSONモードの例"""
    print("\n=== JSONモードの例 ===")
    client = LlamaClient()

    try:
        process = client.json_session()

        # JSONリクエストを送信
        requests = [
            {
                "prompt": "おはようございます",
                "max_tokens": 50,
                "temperature": 0.5
            },
            {
                "prompt": "AIについて教えて",
                "max_tokens": 100,
                "temperature": 0.7
            }
        ]

        for req in requests:
            print(f"送信: {req}")

            process.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            process.stdin.flush()

            # 応答を読み取り
            response_line = process.stdout.readline()
            response_data = json.loads(response_line)

            if response_data['status'] == 'success':
                print(f"応答: {response_data['response']}")
            else:
                print(f"エラー: {response_data['error']}")

        # 終了
        process.stdin.write("EXIT\n")
        process.stdin.flush()
        process.wait()

    except Exception as e:
        print(f"エラー: {e}")

def main():
    print("Llama Subprocess Client Examples")
    print("=" * 50)

    # 使用例を実行
    example_single()
    example_interactive()
    example_json()

if __name__ == "__main__":
    main()