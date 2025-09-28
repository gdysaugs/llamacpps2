# 🎭 SoVITS-Wav2Lip-LlamaCPP統合システム Docker版

**AI音声合成・リップシンク・顔交換・AI会話** 完全統合システム - Docker版ガイド

## 🚀 クイックスタート

### 必要要件
- **Docker** (20.10+)
- **NVIDIA Docker Runtime** (nvidia-docker2)
- **NVIDIA GPU** (RTX 3060以上推奨)
- **GPU メモリ**: 8GB以上
- **ディスク容量**: 50GB以上

### 1. 前提セットアップ

```bash
# NVIDIA Docker Runtimeインストール (Ubuntu/WSL2)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. イメージビルド（初回のみ）

```bash
# プロジェクトディレクトリに移動
cd /home/adama/wav2lip-project

# Docker イメージをビルド（20-30分）
docker build -f Dockerfile.gcr-integrated -t wav2lip-gpu-models:latest .
```

### 3. コンテナ起動

```bash
# GPUサポート付きでコンテナ起動
docker run --gpus all -p 8080:8080 -d wav2lip-gpu-models:latest

# または、名前付きで起動
docker run --gpus all -p 8080:8080 --name wav2lip-container -d wav2lip-gpu-models:latest
```

### 4. アクセス

- **ローカルアクセス**: http://localhost:8080
- **WSL2**: http://localhost:8080 （Windowsブラウザから）

## 🔧 Docker管理コマンド

### コンテナ操作
```bash
# 実行中コンテナ確認
docker ps

# コンテナ停止
docker stop wav2lip-container

# コンテナ削除
docker rm wav2lip-container

# コンテナログ確認
docker logs wav2lip-container

# コンテナ内でシェル実行
docker exec -it wav2lip-container bash
```

### イメージ管理
```bash
# イメージ一覧
docker images | grep wav2lip

# 古いイメージ削除
docker rmi wav2lip-gpu-models:latest

# イメージ再ビルド（強制）
docker build --no-cache -f Dockerfile.gcr-integrated -t wav2lip-gpu-models:latest .
```

## 🎯 使用方法

### 基本ワークフロー

1. **ファイル準備**
   - 動画ファイル (MP4, AVI, MOV対応)
   - リファレンス音声 (WAV, MP3対応)
   - セリフテキスト

2. **処理実行**
   - Phase 1: SoVITS音声生成
   - Phase 2: Wav2Lipリップシンク
   - Phase 3: FaceFusion顔交換（オプション）

3. **結果確認**
   - 生成された動画ダウンロード
   - ログで処理詳細確認

### AI会話モード

```
✅ AI会話モード ON
└── LlamaCPP (Berghof-NSFW-7B) が応答生成
└── キャラクター特徴でプロンプト調整可能
└── 自然な日本語での対話
```

## 🛠️ トラブルシューティング

### 1. SoVITSパスエラー

**症状**: `FileNotFoundError: GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth`

**原因**: モデルファイルのシンボリックリンクが壊れている

**解決方法**:
```bash
# コンテナ内でパス修正
docker exec -it <CONTAINER_ID> bash

# s2Gv4.pthリンク作成
ln -sf /app/gpt_sovits_full/GPT_SoVITS/pretrained_models/s2Gv4.pth \
       /app/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth

# vocoder.pthリンク修正
rm -f /app/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth
ln -sf /app/gpt_sovits_full/GPT_SoVITS/pretrained_models/gpt_sovits_models_vocoder.pth \
       /app/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth

# コンテナ再起動
exit
docker restart <CONTAINER_ID>
```

### 2. ポート競合エラー

**症状**: `port is already allocated`

**解決方法**:
```bash
# 別のポートで起動
docker run --gpus all -p 8081:8080 -d wav2lip-gpu-models:latest

# または既存コンテナ停止
docker ps | grep wav2lip
docker stop <CONTAINER_ID>
```

### 3. GPU認識エラー

**症状**: `RuntimeError: No CUDA GPUs are available`

**解決方法**:
```bash
# NVIDIA Docker Runtime確認
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# CPUモードで起動（非推奨）
docker run -p 8080:8080 -e CUDA_VISIBLE_DEVICES="" -d wav2lip-gpu-models:latest
```

### 4. メモリ不足エラー

**症状**: `CUDA out of memory`

**解決方法**:
```bash
# GPU メモリ制限設定
docker run --gpus all --memory=16g --memory-swap=32g \
  -p 8080:8080 -d wav2lip-gpu-models:latest

# 処理前にGPUメモリクリア
docker exec <CONTAINER_ID> python -c "import torch; torch.cuda.empty_cache()"
```

### 5. モデルファイル不足

**症状**: 各種 `FileNotFoundError`

**解決方法**:
```bash
# コンテナ内モデル確認
docker exec <CONTAINER_ID> find /app -name "*.pth" -o -name "*.ckpt" -o -name "*.gguf"

# 不足している場合は再ビルド
docker build --no-cache -f Dockerfile.gcr-integrated -t wav2lip-gpu-models:latest .
```

## 📊 パフォーマンス調整

### 最適化設定

```bash
# 高性能GPU (RTX 4090等)
docker run --gpus all -p 8080:8080 \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024 \
  -d wav2lip-gpu-models:latest

# 低メモリGPU (RTX 3060等)
docker run --gpus all -p 8080:8080 \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
  -d wav2lip-gpu-models:latest
```

### 処理時間目安

| GPU | Phase 1 (SoVITS) | Phase 2 (Wav2Lip) | 合計 |
|-----|-------------------|-------------------|------|
| RTX 4090 | 15-30秒 | 30-60秒 | 1-2分 |
| RTX 3080 | 30-45秒 | 60-90秒 | 2-3分 |
| RTX 3060 | 45-60秒 | 90-120秒 | 3-4分 |

## 🔍 ログ解析

### コンテナログ確認
```bash
# リアルタイムログ
docker logs -f <CONTAINER_ID>

# 最新100行
docker logs --tail 100 <CONTAINER_ID>

# エラーのみ抽出
docker logs <CONTAINER_ID> 2>&1 | grep -i error
```

### 処理ログパターン
```
✅ 正常起動:
🎭🎬🤖 SOVITS-Wav2Lip-LlamaCPP Integration System
✅ システム初期化完了
🌐 アクセスURL: http://0.0.0.0:8080

✅ 正常処理:
🚀 ポータブル版統合パイプライン開始
🎵 Phase 1: SoVITS音声生成開始
✅ Phase 1完了: /app/output/temp_*.wav
🎬 Phase 2: Wav2Lip リップシンク開始
✅ Phase 2完了: /app/temp/temp_*.mp4

❌ エラーパターン:
❌ Phase 1失敗: ❌ SOVITS Voice Cloning Failed
🔍 Error reason: Process failed with return code 1
```

## 📁 ファイル構造（Docker版）

```
/app/ (コンテナ内)
├── gpt_sovits_full/          # GPT-SoVITSフルバージョン
│   └── GPT_SoVITS/
│       └── pretrained_models/
│           ├── s2Gv4.pth         # VITSモデル (769MB)
│           ├── gpt_sovits_models_vocoder.pth # Vocoder (57MB)
│           ├── chinese-hubert-base/
│           └── chinese-roberta-wwm-ext-large/
├── models/                    # 追加モデル
│   ├── gpt_sovits/
│   ├── wav2lip/
│   └── facefusion/
├── gradio_frontend/           # Webインターフェース
│   ├── wav2lip_sovits_llama_integrated_portable.py
│   └── sovits_wav2lip_integration.py
├── checkpoints/               # Wav2Lipモデル
├── output/                    # 生成結果
└── temp/                      # 一時ファイル
```

## 🚧 開発・カスタマイズ

### カスタムビルド
```bash
# 開発用マウント起動
docker run --gpus all -p 8080:8080 \
  -v $(pwd)/gradio_frontend:/app/gradio_frontend:ro \
  -d wav2lip-gpu-models:latest

# コードホットリロード
docker exec <CONTAINER_ID> pkill -f "wav2lip_sovits_llama_integrated_portable.py"
```

### 環境変数設定
```bash
# カスタム設定で起動
docker run --gpus all -p 8080:8080 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  -e GRADIO_SERVER_NAME=0.0.0.0 \
  -e GRADIO_SERVER_PORT=8080 \
  -d wav2lip-gpu-models:latest
```

## ⚠️ 注意事項

- **GPU必須**: CPU-onlyモードは実用的ではありません
- **VRAM**: 最低8GB、推奨12GB以上
- **処理時間**: 1分の動画で3-5分の処理時間
- **品質**: 高品質モードは処理時間が2-3倍
- **ネットワーク**: 初回ビルド時に大量データダウンロード

## 📞 サポート

### ログ提出時の情報
```bash
# システム情報
nvidia-smi
docker --version
docker info | grep -i runtime

# コンテナ情報
docker inspect <CONTAINER_ID>
docker logs --tail 200 <CONTAINER_ID>
```

---

**最終更新**: 2025-09-26
**バージョン**: 2.0-docker (Docker統合版)
**作成者**: Claude Code Assistant