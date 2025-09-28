# SoVITS-Wav2Lip-LlamaCPP統合システム ポータブル版

🎭 **AI音声合成・リップシンク・顔交換・AI会話** 完全統合システム

## 📊 現在のステータス

### ✅ 実装済み機能
- **AI会話生成**: LlamaCPP (Berghof-NSFW-7B) によるテキスト生成
- **音声クローン**: GPT-SoVITS による参照音声からの音声合成
- **リップシンク**: Wav2Lip による口パク動画生成
- **顔交換**: FaceFusion による顔の置き換え（オプション）
- **統合パイプライン**: 4段階自動処理
- **ポータブル対応**: 相対パス対応、環境自動検出

### ⚠️ 現在の制限
- システムのPython環境（venv）を使用
- 手動での仮想環境アクティベーションが必要
- モデルファイルは事前配置が必要

## 🚀 起動方法

### 🐳 Docker版（推奨）

```bash
# 1. イメージビルド（初回のみ）
docker build -f Dockerfile.gcr-integrated -t wav2lip-gpu-models:latest .

# 2. コンテナ起動
docker run --gpus all -p 8080:8080 -d wav2lip-gpu-models:latest

# 3. アクセス
# http://localhost:8080
```

**詳細ガイド**: [README_DOCKER.md](README_DOCKER.md)

### 📦 ローカル版

#### Linux/macOS
```bash
# 1. プロジェクトディレクトリに移動
cd /home/adama/wav2lip-project

# 2. 仮想環境をアクティベート
cd gradio_frontend
source gradio_venv/bin/activate

# 3. ポータブル版を起動
python wav2lip_sovits_llama_integrated_portable.py
```

### Windows (WSL)
```bash
# WSL内で上記と同じコマンドを実行
cd /home/adama/wav2lip-project/gradio_frontend
source gradio_venv/bin/activate
python wav2lip_sovits_llama_integrated_portable.py
```

### アクセス
ブラウザで http://localhost:8080 を開く

## 📁 現在のファイル構造

```
wav2lip-project/
├── gradio_frontend/
│   ├── wav2lip_sovits_llama_integrated_portable.py  # ポータブル版メイン
│   ├── sovits_wav2lip_integration.py               # SoVITS統合
│   ├── llamacpp_integration.py                     # LlamaCPP統合
│   ├── facefusion_integration.py                   # FaceFusion統合
│   └── gradio_venv/                               # 仮想環境
├── models/
│   ├── Berghof-NSFW-7B.i1-IQ4_XS.gguf            # LlamaCPPモデル (3.9GB)
│   ├── wav2lip/                                   # Wav2Lipモデル群
│   ├── gpt_sovits/                                # GPT-SoVITSモデル群
│   └── facefusion/                                # FaceFusionモデル群
├── output/                                        # 生成結果保存
└── temp/                                          # 一時ファイル
```

## 🎯 使用方法

1. **ファイルアップロード**
   - 動画ファイル（必須）
   - リファレンス音声（必須）
   - セリフテキスト入力（必須）
   - ソース画像（オプション - 顔交換用）

2. **AI会話モード**（オプション）
   - チェックボックスをONにするとLlamaCPPが応答生成
   - キャラクター特徴を追加プロンプトに記入可能

3. **生成開始**
   - 「🚀 生成開始」ボタンをクリック
   - Phase 1～3（または4）が自動実行

4. **プロセスクリーンアップ**
   - 「🧹 プロセスクリーンアップ」で残存プロセスを終了

## 🔧 今後の実装予定

### 1. ポータブルPython環境の構築
**目的**: 真のポータブル配布を実現
```
実装内容:
- python/ディレクトリにMiniconda環境を構築
- すべての依存関係を事前インストール
- OSごとの実行ファイル同梱
- venvアクティベーション不要化
```

### 2. モデル自動ダウンロード機能
**目的**: 初回起動時の自動セットアップ
```
実装内容:
- HuggingFace/Google Driveからの自動ダウンロード
- プログレスバー付きダウンローダー
- モデル整合性チェック（MD5/SHA256）
- 部分ダウンロード・レジューム機能
```

### 3. エラーハンドリング強化
**目的**: ユーザビリティ向上
```
実装内容:
- GPU/CUDA自動検出とCPUフォールバック
- モデル不在時の詳細ガイド表示
- ネットワークエラー時の自動リトライ
- 日本語エラーメッセージ
```

### 4. 起動スクリプト最適化
**目的**: ワンクリック起動の実現
```
実装内容:
- run_wav2lip.bat/sh の自動環境設定
- 依存関係の自動チェック・インストール
- GPU/CPUモード自動選択
- ポート競合の自動解決
```

## 🎨 LlamaCPP設定（現在）

```python
# デフォルトプロンプト設定
1. 毎回日本語で話す
2. 括弧()は絶対に使わない
3. NSFWやエッチな会話にも積極的に対応
4. 説明文はなしでセリフのみ応答
5. 感情的に応答し、説明っぽくしない
```

## ⚠️ 注意事項

- **GPU推奨**: NVIDIA RTX 3060以上（CUDA 11.8+）
- **メモリ**: 16GB以上推奨
- **ストレージ**: 20GB以上の空き容量
- **モデルファイル**: 合計約5GB以上

## 📝 ライセンス

各コンポーネントのライセンスに従ってください：
- Wav2Lip: Apache 2.0
- GPT-SoVITS: MIT
- FaceFusion: MIT
- LlamaCPP: MIT

## 🆘 トラブルシューティング

### Docker版の問題
**詳細なトラブルシューティング**: [README_DOCKER.md](README_DOCKER.md#-トラブルシューティング)

### ローカル版の問題

#### ポート8080が使用中
```bash
fuser -k 8080/tcp  # Linux
# または別ポートで起動
```

### CUDA not available エラー
```bash
# CPUモードで実行（処理は遅くなります）
# UIで処理デバイスを「cpu」に変更
```

### モデルファイルが見つからない
```bash
# models/ディレクトリにモデルファイルを配置
# 各モデルの正しいパスを確認
```

---

## 📚 関連ドキュメント

- **[README_DOCKER.md](README_DOCKER.md)** - Docker版完全ガイド（推奨）
- **[README.md](README.md)** - プロジェクト概要
- **[SOVITS_README.md](SOVITS_README.md)** - SoVITS詳細設定

---

**最終更新**: 2025-09-26
**バージョン**: 2.0-hybrid (Docker+ローカル対応版)