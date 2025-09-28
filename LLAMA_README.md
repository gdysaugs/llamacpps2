# Llama.cpp Python - ローカルLLM実行環境

## 概要
このプロジェクトはllama-cpp-pythonを使用してローカルでLLMモデルを実行するための環境です。
GPU（CUDA）アクセラレーションに対応しており、高速な推論が可能です。

## システム要件
- Python 3.10以上
- CUDA 12.1以上（GPU使用時）
- 8GB以上のRAM（推奨：16GB以上）
- 4GB以上のVRAM（GPU使用時、推奨：8GB以上）

## インストール済み環境
- **仮想環境**: `/home/adama/wav2lip-project/llama_venv`
- **Pythonバージョン**: 3.10.12
- **llama-cpp-python**: 0.3.16（CUDA 12.1対応版）

## 使用モデル
- **モデル名**: Berghof-NSFW-7B.i1-IQ4_XS.gguf
- **モデルパス**: `/home/adama/wav2lip-project/models/Berghof-NSFW-7B.i1-IQ4_XS.gguf`
- **モデルサイズ**: 7B parameters（IQ4_XS量子化）
- **コンテキストサイズ**: 4096トークン

## 起動方法

### 1. 仮想環境の有効化
```bash
cd /home/adama/wav2lip-project
source llama_venv/bin/activate
```

### 2. インタラクティブ会話モード
```bash
python test_llama_cli.py
```

#### コマンド
- `exit` または `quit`: 終了
- `clear`: 会話履歴をクリア
- Ctrl+C: 現在の生成を中断

### 3. シンプルテスト実行
```bash
python test_llama_simple.py
```

## プログラム仕様

### test_llama_cli.py - インタラクティブ会話用
**主な機能:**
- リアルタイムストリーミング応答
- 会話履歴の保持（最大10交換）
- システムプロンプトによる応答品質向上
- 日本語・英語両対応

**パラメータ設定:**
```python
n_ctx=4096          # コンテキストウィンドウサイズ
n_gpu_layers=35     # GPU使用レイヤー数
n_threads=4         # CPU使用スレッド数
max_tokens=1024     # 最大生成トークン数
temperature=0.7     # 生成温度（創造性）
top_p=0.95         # nucleus sampling
top_k=40           # top-k sampling
repeat_penalty=1.1  # 繰り返しペナルティ
```

### test_llama_simple.py - 動作確認用
**主な機能:**
- モデルロードテスト
- GPU/CPU自動切り替え
- 簡単な質問応答テスト

**パラメータ設定:**
```python
n_ctx=2048         # コンテキストウィンドウサイズ
n_gpu_layers=35    # GPU使用レイヤー数（GPU時）
n_gpu_layers=0     # CPU専用モード（フォールバック）
max_tokens=100     # 最大生成トークン数
temperature=0.7    # 生成温度
```

## Python APIの使用例

### 基本的な使用方法
```python
from llama_cpp import Llama

# モデルのロード
llm = Llama(
    model_path="/home/adama/wav2lip-project/models/Berghof-NSFW-7B.i1-IQ4_XS.gguf",
    n_ctx=2048,
    n_gpu_layers=35,  # GPUアクセラレーション
    verbose=False
)

# テキスト生成
response = llm(
    "Human: こんにちは\n\nAssistant:",
    max_tokens=100,
    temperature=0.7,
    stop=["Human:", "\n\n"]
)

print(response['choices'][0]['text'])
```

### ストリーミング生成
```python
# ストリーミングレスポンス
for chunk in llm(prompt, max_tokens=100, stream=True, echo=False):
    print(chunk['choices'][0]['text'], end='', flush=True)
```

### チャット形式
```python
# 会話履歴を使用
messages = [
    {"role": "system", "content": "あなたは親切なアシスタントです。"},
    {"role": "user", "content": "天気について教えて"},
]

response = llm.create_chat_completion(
    messages=messages,
    max_tokens=512,
    temperature=0.7,
    stream=True
)

for chunk in response:
    delta = chunk['choices'][0]['delta']
    if 'content' in delta:
        print(delta['content'], end='', flush=True)
```

## トラブルシューティング

### GPU が認識されない場合
```bash
# CUDA環境変数の設定
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### メモリ不足エラー
- `n_gpu_layers`の値を減らす（例：35→20）
- `n_ctx`の値を減らす（例：4096→2048）
- CPU専用モードを使用（`n_gpu_layers=0`）

### 生成が遅い場合
- GPU使用を確認（`n_gpu_layers`が0以上）
- `n_threads`を増やす（CPUコア数まで）
- モデルの量子化レベルを確認

## パフォーマンス指標
- **GPU使用時**: 約30-50 tokens/秒
- **CPU使用時**: 約5-10 tokens/秒
- **初回ロード時間**: 約5-10秒

## 依存関係
```
llama-cpp-python==0.3.16
numpy>=1.20.0
typing-extensions>=4.5.0
diskcache>=5.6.1
jinja2>=2.11.3
```

## 注意事項
- モデルは成人向けコンテンツを含む可能性があります
- 商用利用の際はモデルのライセンスを確認してください
- 長時間の使用時はGPUの温度に注意してください

## 更新履歴
- 2025-09-15: 初期セットアップ完了
- llama-cpp-python 0.3.16 (CUDA 12.1対応版)インストール
- テストスクリプト作成