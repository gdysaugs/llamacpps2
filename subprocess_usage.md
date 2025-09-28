# Llama Subprocess使用ガイド

## 概要
サブプロセスでLlamaモデルを呼び出すためのインターフェースです。
他のプログラムからLlamaを簡単に使用できます。

## ファイル構成
- `llama_subprocess.py` - サブプロセス用メインスクリプト
- `llama_client_example.py` - クライアント使用例
- `subprocess_usage.md` - このガイド

## 使用方法

### 1. 単発生成モード（コマンドライン）
```bash
source llama_venv/bin/activate
python llama_subprocess.py --mode single --prompt "こんにちは" --max_tokens 100
```

### 2. Pythonからサブプロセス呼び出し

#### 単発生成
```python
import subprocess

def call_llama(prompt):
    cmd = [
        "/home/adama/wav2lip-project/llama_venv/bin/python",
        "/home/adama/wav2lip-project/llama_subprocess.py",
        "--mode", "single",
        "--prompt", prompt,
        "--max_tokens", "200"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

# 使用例
response = call_llama("今日の天気はどうですか？")
print(response)
```

#### インタラクティブセッション
```python
import subprocess

def interactive_llama():
    cmd = [
        "/home/adama/wav2lip-project/llama_venv/bin/python",
        "/home/adama/wav2lip-project/llama_subprocess.py",
        "--mode", "interactive"
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )

    # プロンプト送信
    process.stdin.write("こんにちは\n")
    process.stdin.flush()

    # 応答読み取り
    while True:
        line = process.stdout.readline()
        if line.startswith("RESPONSE:"):
            response = line[9:].strip()
            print(f"応答: {response}")
        elif line.strip() == "END_RESPONSE":
            break

    # 終了
    process.stdin.write("EXIT\n")
    process.stdin.flush()
    process.wait()

interactive_llama()
```

#### JSONモード
```python
import subprocess
import json

def json_llama():
    cmd = [
        "/home/adama/wav2lip-project/llama_venv/bin/python",
        "/home/adama/wav2lip-project/llama_subprocess.py",
        "--mode", "json"
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )

    # JSONリクエスト
    request = {
        "prompt": "AIについて説明して",
        "max_tokens": 150,
        "temperature": 0.7
    }

    process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
    process.stdin.flush()

    # JSON応答
    response_line = process.stdout.readline()
    response_data = json.loads(response_line)

    if response_data['status'] == 'success':
        print(response_data['response'])

    # 終了
    process.stdin.write("EXIT\n")
    process.wait()

json_llama()
```

## コマンドライン引数

### 基本オプション
- `--mode`: 実行モード (`single`/`interactive`/`json`)
- `--prompt`: 入力プロンプト（singleモード用）
- `--max_tokens`: 最大生成トークン数（デフォルト: 512）
- `--temperature`: 生成温度（デフォルト: 0.7）
- `--model_path`: モデルファイルパス（オプション）

### 使用例
```bash
# 基本使用
python llama_subprocess.py --mode single --prompt "Hello"

# パラメータ指定
python llama_subprocess.py --mode single --prompt "Hello" --max_tokens 200 --temperature 0.5

# インタラクティブモード
python llama_subprocess.py --mode interactive

# JSONモード
python llama_subprocess.py --mode json
```

## 実行例とテスト

### テストスクリプト実行
```bash
source llama_venv/bin/activate
python llama_client_example.py
```

### クライアントクラス使用例
```python
from llama_client_example import LlamaClient

client = LlamaClient()

# 単発生成
response = client.single_generation("今日はいい天気ですね")
print(response)

# インタラクティブセッション
process = client.interactive_session()
# ... プロセス操作

# JSONセッション
process = client.json_session()
# ... JSON通信
```

## パフォーマンス特性
- 初回ロード: 約2-3秒（GPU全レイヤー使用）
- 生成速度: 約10-15 tokens/秒
- メモリ使用: GPU 3.6GB、RAM 1-2GB

## エラーハンドリング
- モデルファイルが見つからない場合は`FileNotFoundError`
- JSON解析エラーはエラー応答を返す
- プロセス終了は`EXIT`コマンドまたは`Ctrl+C`

## 他言語からの呼び出し

### bash/shell
```bash
#!/bin/bash
source /home/adama/wav2lip-project/llama_venv/bin/activate
response=$(python /home/adama/wav2lip-project/llama_subprocess.py --mode single --prompt "$1")
echo "$response"
```

### Node.js
```javascript
const { spawn } = require('child_process');

function callLlama(prompt) {
    return new Promise((resolve, reject) => {
        const python = spawn('/home/adama/wav2lip-project/llama_venv/bin/python', [
            '/home/adama/wav2lip-project/llama_subprocess.py',
            '--mode', 'single',
            '--prompt', prompt
        ]);

        let output = '';
        python.stdout.on('data', (data) => {
            output += data.toString();
        });

        python.on('close', (code) => {
            if (code === 0) {
                resolve(output.trim());
            } else {
                reject(new Error(`Process exited with code ${code}`));
            }
        });
    });
}

// 使用例
callLlama('Hello, how are you?').then(response => {
    console.log(response);
});
```

## 注意事項
- 初回実行時はモデルロードに時間がかかります
- GPU メモリを大量に使用するため、他のGPU処理と並行実行は避けてください
- 長時間のセッションではメモリリークに注意してください
- プロセス終了時は必ず`EXIT`を送信するか適切にプロセスを終了してください