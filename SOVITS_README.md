# GPT-SoVITS v4 高品質音声クローンシステム

## 🚀 概要
GPT-SoVITS v4を使用した最新の高品質音声クローンシステム。任意音声形式の自動変換・3秒未満音声の自動延長・極限感情パラメータによる自然な音声生成を実現。

## ✨ 主要機能
- 🎵 **全音声形式対応**: MP3/M4A/FLAC/OGG等を自動でWAV変換
- ⏱ **短時間音声自動延長**: 3秒未満の音声を自動で3秒に延長
- 🎭 **極限感情表現**: temperature=2.0による非常に感情豊かな音声生成
- 🧠 **v4モデル対応**: 最新のGPT-SoVITS v4モデルで最高品質
- 🔧 **完全自動化**: 前処理から生成まで全て自動実行

## 📝 基本使用方法

### 1. 標準的な使用方法
```bash
# 仮想環境アクティベート
source gpt_sovits_env/bin/activate

# 音声クローン生成（任意の音声形式・長さに対応）
# 注：出力ファイルは自動的に output/ フォルダに保存されます
python gpt_sovits_simple_cli.py "ソース音声.mp3" "生成したいテキスト" "ファイル名.wav"
```

### 2. 実行例

#### MP3からの感情的音声生成
```bash
python gpt_sovits_simple_cli.py "/path/to/voice.mp3" "こんにちは、今日はとても良い天気ですね" "hello.wav"
```

#### 短い音声からの長文生成
```bash
python gpt_sovits_simple_cli.py "short_voice.wav" "やめて！そんなことしないで！お願いだから...いやあああ！" "emotional.wav"
```

## 🎯 自動処理機能

### 1. 音声形式の自動変換
システムが自動で以下の処理を実行：
- **MP3/M4A/FLAC/OGG** → **WAV (44.1kHz mono)** に自動変換
- 変換ファイルは `temp_audio/` フォルダに保存・再利用
- ffmpegを使用した高品質変換

### 2. 短時間音声の自動延長
- **3秒未満の音声**: 自動で3秒に延長（パディング）
- **v4モデル要求**: 3秒以上の音声が必要なため
- サイレント部分で自然に延長

### 3. 処理ログ例
```
📄 音声形式変換中: .mp3 -> WAV
✅ WAV変換完了: /temp_audio/voice_converted.wav
🎵 音声長さ: 1.01秒
⏱ 音声が短いため3秒に延長中...
✅ 音声延長完了: /temp_audio/voice_converted_extended.wav (3.00秒)
```

## ⚙️ 感情パラメータ設定

### 極限感情表現設定（現在の設定）
```python
request_params = {
    "top_k": 5,           # 少数精鋭で感情的な候補選択
    "top_p": 0.75,        # 多様性を高めて感情表現豊かに
    "temperature": 2.0,    # 感情表現を非常に豊かに（極限設定）
    "speed_factor": 1.1,   # 少し速めの自然な話速
    "repetition_penalty": 1.2,  # 繰り返しを少し許容
    "sample_steps": 48,    # 高品質化
    "ref_free": False      # v4ではprompt_textを使用
}
```

### パラメータ解説
- **temperature=2.0**: 極限の感情表現（棒読み完全回避）
- **top_k=5**: 感情的で劇的な候補のみ選択
- **top_p=0.75**: 表現の多様性を大幅向上
- **prompt_text**: ターゲットテキストと同一設定（v4要求）

## 🎭 対応する感情表現

### 1. 基本感情
```bash
# 挨拶・日常会話
"こんにちは、今日はとても良い天気ですね。お元気でいらっしゃいますか？"

# 短い表現
"こん"  # 2文字でも自然な音声生成
```

### 2. 感情的表現
```bash
# 嫌がり・抵抗表現
"やめて！そんなことしないで！お願いだから...いやあああ！だめ、だめよ！"

# 喘ぎ声・感情的表現
"あっ、んっ...だめ...そんなに激しくしたら...はぁはぁ...気持ちよくなっちゃう..."
```

## 📊 実証済み性能

### テスト結果（実測値）
| テキスト長 | 処理時間 | 出力時間 | ファイルサイズ | 品質 |
|------------|----------|----------|----------------|------|
| 2文字「こん」 | ~8秒 | 1.82秒 | 0.17MB | ⭐⭐⭐⭐⭐ |
| 25文字（日常） | ~15秒 | 6.38秒 | 0.58MB | ⭐⭐⭐⭐⭐ |
| 60文字（感情的） | ~50秒 | 16.88秒 | 1.55MB | ⭐⭐⭐⭐⭐ |
| 80文字（長文感情） | ~60秒 | 19.62秒 | 1.80MB | ⭐⭐⭐⭐⭐ |

## 🔧 システム構成

### ファイル構成
```
wav2lip-project/
├── gpt_sovits_simple_cli.py      # メイン実行スクリプト
├── gpt_sovits_env/              # Python仮想環境
├── models/gpt_sovits/           # モデル・音声ファイル
│   ├── gpt_sovits_model.ckpt    # カスタムGPTモデル
│   └── pretrained_models/       # 事前学習モデル
│       └── s2Gv4.pth           # v4 VITSモデル
├── temp_audio/                  # 自動変換音声の保存先
└── output/                      # 生成音声の出力先（全ての出力ファイルはここに保存）
```

### 依存関係
- Python 3.10+
- CUDA対応GPU（推奨）
- ffmpeg（音声変換用）
- librosa（音声解析用）
- torch, soundfile等（requirements.txt参照）

## 🚀 最適化された機能

### 1. v4モデル対応
- **最新モデル**: GPT-SoVITS v4対応
- **prompt_text**: ターゲットテキストと同一に自動設定
- **並列推論**: 高速生成モード有効

### 2. 自動化システム
- **ファイル検証**: 存在確認・形式チェック
- **自動変換**: 非WAVファイルの自動変換
- **自動延長**: 3秒未満音声の自動パディング
- **エラー処理**: 詳細ログ・エラー回復

### 3. メモリ効率
- **GPU半精度**: CUDA環境で自動有効
- **一時ファイル**: 適切な管理・クリーンアップ
- **プロセス最適化**: 効率的なリソース使用

## 💡 使用のコツ

### 1. 音声ソース選択
- **クリアな音声**: ノイズの少ない音声が最適
- **任意の長さ**: 1秒でも30秒でも自動対応
- **任意の形式**: MP3でもM4AでもOK

### 2. テキスト入力
- **感情表現**: 「！」「...」「んっ」等の表現が効果的
- **長文対応**: 自動で文章分割処理
- **特殊表現**: 喘ぎ声・叫び声も自然に生成

### 3. 出力最適化
- **高品質設定**: sample_steps=48で最高品質
- **自然な速度**: speed_factor=1.1で自然な話速
- **感情豊か**: temperature=2.0で非常に表現豊か

---

## 📝 実行コマンドまとめ

### 基本実行方法

```bash
# 基本実行
source gpt_sovits_env/bin/activate
python gpt_sovits_simple_cli.py "音声ファイル" "生成テキスト" "出力ファイル名"
# 注：全ての出力ファイルは自動的に output/ フォルダに保存されます

# MP3からの生成（output/hello.wav に保存）
python gpt_sovits_simple_cli.py "/path/to/voice.mp3" "こんにちは" "hello.wav"

# 感情的な長文生成（output/emotional.wav に保存）
python gpt_sovits_simple_cli.py "voice.wav" "やめて！だめ...んっ...あああ！" "emotional.wav"
```

---

## 🔧 サブプロセス実行（他のPythonスクリプトから呼び出す場合）

### サブプロセス実行の利点
- **独立実行**: 仮想環境を別プロセスで実行（メモリ分離）
- **安定性**: GPUメモリを効率的に管理
- **互換性**: 異なるPython環境から安全に呼び出し可能

### Python コード例

```python
import subprocess
from pathlib import Path

def generate_voice_clone(ref_audio, target_text, output_file):
    """
    GPT-SoVITSをサブプロセスで実行して音声生成

    Args:
        ref_audio: 参照音声ファイルパス
        target_text: 生成したいテキスト
        output_file: 出力ファイル名（output/に保存される）

    Returns:
        bool: 成功時True
    """
    project_root = Path("/home/adama/wav2lip-project")
    venv_python = project_root / "gpt_sovits_env/bin/python"
    cli_script = project_root / "gpt_sovits_simple_cli.py"

    cmd = [
        str(venv_python),
        str(cli_script),
        ref_audio,
        target_text,
        output_file
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
            check=True
        )
        print(f"✅ 音声生成成功: output/{output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ エラー: {e.stderr}")
        return False

# 使用例
generate_voice_clone(
    "models/gpt_sovits/baka_new_reference.wav",
    "サブプロセスから音声生成テストです",
    "subprocess_output.wav"
)
```

### テストスクリプト実行

```bash
# サブプロセステストスクリプトを実行
python3 test_subprocess_sovits.py
```

### 実行結果例
```
============================================================
GPT-SoVITS サブプロセステスト開始
============================================================
📂 作業ディレクトリ: /home/adama/wav2lip-project
🐍 Python: gpt_sovits_env/bin/python
🎤 参照音声: models/gpt_sovits/baka_new_reference.wav
📝 生成テキスト: サブプロセステストです。これは正常に動作しています。
💾 出力ファイル: output/subprocess_test.wav
------------------------------------------------------------
✅ サブプロセス実行成功！
⏱ 実行時間: 42.63秒
📊 ファイルサイズ: 0.37MB
============================================================
```

### 注意事項
- **メモリ管理**: 各実行で約4-8GB のGPUメモリを使用
- **処理時間**: 初回実行時はモデル読み込みで約20秒、音声生成に10-40秒
- **並列実行**: 複数同時実行は避ける（GPUメモリ制限のため）

---

🎯 **結論**: 任意音声形式・任意長さ・任意テキストから、極限の感情表現による最高品質な音声クローンを自動生成可能。v4モデルによる最新技術で、棒読みを完全に排除した自然で感情豊かな音声を実現。サブプロセス実行により、様々なPython環境から安全に呼び出し可能。