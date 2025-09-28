# SoVITS-Wav2Lip 統合システム

音声クローン生成 + リップシンクの2段階統合パイプライン。GPT-SoVITS音声生成、RetinaFace GPU検出、GFPGAN顔補正、onnxruntime-gpu 1.15.1最適化対応。

## 特徴

- 🎵 **GPT-SoVITS音声クローン**: リファレンス音声から任意テキストの音声生成
- 💋 **Wav2Lip リップシンク**: 高精度リップシンク動画生成
- 🎯 **RetinaFace GPU検出**: 高精度顔検出でリップシンク品質向上
- 🖼️ **GFPGAN顔補正**: GPU加速による高品質顔画質向上
- ⚡ **GPU最適化**: onnxruntime-gpu 1.15.1 + CUDNN v8で最高速度
- 🚀 **2段階パイプライン**: SoVITS音声生成 → Wav2Lipリップシンク
- 🔄 **サブプロセス実行**: メモリ分離による安定実行
- 🌐 **Webフロントエンド**: Gradio搭載の直感的なWebインターフェース
- 📐 **任意解像度対応**: 元解像度完全保持、品質劣化なし

## 必要環境

- Python 3.10+
- NVIDIA GPU (CUDA 11.x/12.x対応)
- CUDNN v8.x (onnxruntime-gpu 1.15.1互換)
- FFmpeg
- GPT-SoVITS v4
- Wav2Lip ONNX モデル

## セットアップ

### 1. 仮想環境の作成・有効化

```bash
cd wav2lip-project
source venv/bin/activate  # Linux/WSL
# または
venv\Scripts\activate     # Windows
```

### 2. 依存関係のインストール

```bash
# onnxruntime-gpu 1.15.1 (CUDNN v8互換) をインストール
pip install onnxruntime-gpu==1.15.1

# その他の依存関係をインストール
pip install opencv-python tqdm librosa scipy numpy pillow
```

### 3. モデルファイルの配置

以下のモデルファイルを適切なディレクトリに配置：

```
models/
├── wav2lip/
│   └── wav2lip_gan.onnx      # メインモデル（必須）
├── gfpgan/
│   └── GFPGANv1.4.onnx      # GFPGAN顔補正モデル（必須）
└── retinaface/
    └── scrfd_2.5g_bnkps.onnx  # RetinaFace検出モデル（内蔵）
```

## 使用方法

### 基本的な使用方法（サブプロセス版 - 推奨）

```bash
python wav2lip_subprocess.py <動画ファイル> <音声ファイル>
```

**例：**
```bash
# 基本実行（GFPGAN顔補正有効）
python wav2lip_subprocess.py video.mp4 audio.wav

# 出力ファイル指定
python wav2lip_subprocess.py video.mp4 audio.wav -o output.mp4

# GFPGAN無効化（最高速度）
python wav2lip_subprocess.py video.mp4 audio.wav --no-gfpgan

# CPU使用（GPU未対応時）
python wav2lip_subprocess.py video.mp4 audio.wav --device cpu
```

### 直接実行版（上級者向け）

```bash
python wav2lip_retinaface_gpu.py --checkpoint_path models/wav2lip/wav2lip_gan.onnx --face video.mp4 --audio audio.wav --outfile output.mp4 --gfpgan
```

### 全オプション一覧

```bash
# サブプロセス版オプション確認
python wav2lip_subprocess.py --help
```

| オプション | デフォルト | 説明 |
|-----------|------------|------|
| `video_path` | - | 入力動画ファイルパス（必須） |
| `audio_path` | - | 入力音声ファイルパス（必須） |
| `-o, --output` | `output/result_subprocess.mp4` | 出力動画パス |
| `--device` | `cuda` | 処理デバイス（cuda/cpu） |
| `--no-gfpgan` | - | GFPGAN顔補正無効化 |

## 使用例

### 1. 標準実行（GFPGAN顔補正有効 - 推奨）
```bash
python wav2lip_subprocess.py video.mp4 audio.wav
```
**処理時間**: 約31秒 (5.2秒動画) | **品質**: 高

### 2. 高速実行（GFPGAN無効）
```bash
python wav2lip_subprocess.py video.mp4 audio.wav --no-gfpgan
```
**処理時間**: 約20秒 (5.2秒動画) | **品質**: 標準

### 3. Windowsパス対応（WSL環境）
```bash
python wav2lip_subprocess.py "/mnt/c/Users/user/video.mp4" "/mnt/c/Users/user/audio.mp3" -o "output.mp4"
```

### 4. クイックテスト実行
```python
# Pythonから直接実行
from wav2lip_subprocess import quick_test
quick_test()
```

### 5. SoVITS-Wav2Lip統合Webフロントエンド起動
```bash
cd gradio_frontend
source gradio_venv/bin/activate
python wav2lip_sovits_integrated.py
```
アクセス: http://localhost:7864

## 2段階統合パイプライン

### Phase 1: SoVITS音声クローン生成
1. **リファレンス音声解析**: GPT-SoVITS v4による音声特徴抽出
2. **テキスト音響変換**: 入力テキストから音響モデル生成
3. **音声合成**: 音声クローンによる高品質音声生成
4. **音声後処理**: FFmpegによる音声最適化

### Phase 2: Wav2Lipリップシンク
1. **動画・音声読み込み**: FFmpegによる前処理とフレーム抽出
2. **RetinaFace検出**: GPU加速による高精度顔検出 (~76it/s)
3. **顔領域正規化**: wav2lip-onnx-HQ互換の96x96アライン処理
4. **顔マスキング**: 下半分マスク処理（口パク領域特定）
5. **Wav2Lip推論**: GPU加速ONNX推論による口パク生成 (~6.3it/s)
6. **GFPGAN補正**: GPU加速による顔画質向上（オプション）
7. **元解像度復元**: 入力解像度完全保持、品質劣化なし
8. **音声統合**: FFmpegによる高品質H.264エンコード

## トラブルシューティング

### GPU関連エラー
```bash
# GPU利用可能確認
python -c "import onnxruntime; print('CUDA' in onnxruntime.get_available_providers())"

# CPU使用に切り替え
python wav2lip_subprocess.py video.mp4 audio.wav --device cpu
```

### onnxruntime-gpu互換性エラー
```bash
# CUDNN v8対応版インストール（推奨）
pip uninstall onnxruntime-gpu -y
pip install onnxruntime-gpu==1.15.1
```

### 音声形式エラー
サポート形式: MP3, WAV, M4A, AAC
FFmpegで事前変換推奨：
```bash
ffmpeg -i input.audio -ac 1 -ar 44100 output.wav
```

## 開発情報

### プロジェクト構成
```
wav2lip-project/
├── wav2lip_subprocess.py     # サブプロセス版（推奨）
├── wav2lip_retinaface_gpu.py # GPU最適化メインスクリプト
├── utils/                    # ユーティリティ
│   ├── scrfd_2.5g_bnkps.onnx # RetinaFace検出モデル
│   └── audio.py              # 音声処理ユーティリティ
├── gradio_frontend/                      # Webフロントエンド
│   ├── wav2lip_sovits_integrated.py  # SoVITS-Wav2Lip統合Webアプリ（ポート7864）
│   ├── sovits_wav2lip_integration.py # SoVITS-Wav2Lip統合モジュール
│   ├── gradio_venv/                  # 専用仮想環境
│   └── requirements.txt              # Frontend依存関係
├── models/                   # ONNXモデル格納
│   ├── wav2lip/
│   │   └── wav2lip_gan.onnx
│   ├── gfpgan/
│   │   └── GFPGANv1.4.onnx
│   └── retinaface/
│       └── scrfd_2.5g_bnkps.onnx
├── output/                   # 出力ディレクトリ
└── venv/                     # メイン仮想環境
```

### システム要件
- **GPU**: NVIDIA GTX 1060以上（4GB VRAM推奨）
- **RAM**: 8GB以上（16GB推奨）
- **ストレージ**: 3GB以上の空き容量（モデル含む）

### 性能指標
- **Face detection**: ~76 it/s（RetinaFace GPU）
- **Inference**: ~6.3 it/s（Wav2Lip GPU）
- **処理時間**: 約31秒（5.2秒動画、GFPGAN有効）
- **対応解像度**: 任意（元解像度保持）

## ライセンス

このプロジェクトは研究・個人利用目的で開発されています。  
使用するモデル（Wav2Lip, GFPGAN）の各ライセンスに従ってください。

## 更新履歴

### v3.0 (2025-09-15) - SoVITS-Wav2Lip統合システム
- **🎵 GPT-SoVITS v4統合**
  - リファレンス音声からの音声クローン生成
  - テキスト入力による任意音声合成
  - 2段階パイプライン（音声生成 → リップシンク）
- **🌐 統合Webフロントエンド**
  - ポート7864でアクセス可能
  - SoVITS + Wav2Lip一体型処理
  - 進捗表示とエラーハンドリング強化

### v2.0 (2025-09-13) - onnxruntime-gpu 1.15.1最適化版
- **⚡ GPU最適化実装**
  - onnxruntime-gpu 1.15.1 + CUDNN v8.x対応
  - 全モデル（Wav2Lip, RetinaFace, GFPGAN）GPU加速
  - Face detection: ~76it/s, Inference: ~6.3it/s達成
- **🎯 RetinaFace検出統合**
  - 高精度顔検出でリップシンク品質向上
  - GPU最適化によるリアルタイム処理
  - フォールバック機能完全廃止（安定性重視）
- **🔄 サブプロセス実行推奨**
  - `wav2lip_subprocess.py`によるメモリ分離
  - 31秒高速処理（5.2秒動画）実現
  - エラーハンドリングと完全ログ出力
- **🎭 GFPGAN GPU加速**
  - GPU並列処理による顔補正高速化
  - 公式プリプロセッシング統合
  - 品質と速度の最適バランス

### v1.4 (2025-09-13) - Webフロントエンド追加
- Gradio Webフロントエンド統合
- ファイルアップロード・プレビュー機能
- リアルタイム進捗表示

### v1.0 - 初回リリース
- wav2lip-onnx-HQ互換処理パイプライン
- 基本的なGFPGAN顔補正統合

## 技術仕様

### パフォーマンス（onnxruntime-gpu 1.15.1）
- **Face detection**: ~76 it/s（RetinaFace GPU）
- **Wav2Lip inference**: ~6.3 it/s（GPU加速）
- **標準処理時間**: 約31秒（5.2秒動画、GFPGAN有効）
- **高速処理時間**: 約20秒（5.2秒動画、GFPGAN無効）
- **メモリ使用**: 処理中 約2GB VRAM → 処理後 0MB（自動解放）

### 動作確認環境
- **OS**: WSL2 Ubuntu 22.04
- **GPU**: NVIDIA GeForce RTX 3050 Laptop (4GB VRAM)
- **CUDA**: 12.6 + CUDNN v8.9.7
- **onnxruntime-gpu**: 1.15.1 (CUDNN v8互換)
- **Python**: 3.10.12
- **FFmpeg**: 4.4.2（H.264/libx264対応）

## 配布準備状況

### EXE化互換性
- ✅ onnxruntime-gpu 1.15.1 + CUDNN v8で高い互換性
- ✅ サブプロセス実行による安定性確保
- ⚠️ 依存関係サイズ: 約1.5-2GB（ONNXRuntime GPU含む）
- ✅ PyInstaller対応（追加DLL同梱必要）

### 推奨配布方法
```
wav2lip-portable/
├── wav2lip_subprocess.exe   # メイン実行ファイル
├── models/                  # ONNXモデル（必須）
│   ├── wav2lip/
│   │   └── wav2lip_gan.onnx
│   ├── gfpgan/
│   │   └── GFPGANv1.4.onnx
│   └── retinaface/
│       └── scrfd_2.5g_bnkps.onnx
├── bin/
│   └── ffmpeg.exe          # H.264対応FFmpeg同梱
└── README.txt
```

---

**開発者**: Claude Code Assistant
**最終更新**: 2025-09-15 15:36

## 🚀 性能ベンチマーク（v3.0 - SoVITS-Wav2Lip統合システム）

### 2段階パイプライン処理時間
| フェーズ | 処理内容 | 平均処理時間 | 詳細 |
|----------|----------|-------------|------|
| **Phase 1** | SoVITS音声クローン生成 | **20-24秒** | GPT-SoVITS v4による音声合成 |
| **Phase 2** | Wav2Lipリップシンク | **11-30秒** | RetinaFace検出 + GFPGAN補正 |
| **合計** | **統合パイプライン** | **40-55秒** | 5秒動画基準（GFPGAN有効） |

### GPU最適化性能（Wav2Lip部分）
| 処理 | 性能指標 | 使用技術 |
|------|----------|----------|
| Face detection | **76 it/s** | RetinaFace GPU |
| Wav2Lip inference | **6.3 it/s** | ONNX GPU加速 |
| GFPGAN補正 | GPU並列処理 | CUDAExecutionProvider |

**統合システムの利点**:
- ✅ リファレンス音声から任意テキスト音声生成
- ✅ 2段階メモリ分離による安定実行
- ✅ GPU確実利用（プロバイダー自動検証）
- ✅ 任意解像度対応（元画質完全保持）
- ✅ 進捗表示とエラーハンドリング強化
- ✅ onnxruntime-gpu 1.15.1最適化