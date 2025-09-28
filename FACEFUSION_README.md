# FaceFusion Face Swapping System

動画の顔を画像の顔に高品質で置き換えるFaceFusionシステム。GPUアクセラレーション対応で高速処理。

## 特徴

- 🎯 **高精度顔検出**: RetinaFace (320x320) による最適化された顔領域特定
- 🔄 **顔交換技術**: InSwapper 128 FP16 による高速・高品質顔置換
- 💎 **顔品質向上**: GFPGAN 1.4 による顔復元・品質向上
- ⚡ **GPU加速**: CUDA対応で高速処理 (90秒で完了)
- 🔧 **順次処理**: Face Swap → GFPGAN の最適化パイプライン
- 📐 **品質制御**: 出力動画の解像度・エンコーダ設定可能

## 必要環境

- Python 3.10+
- CUDA 12.6対応GPU（推奨）
- FFmpeg
- 4GB以上のVRAM（推奨）

## セットアップ

### 1. 仮想環境の有効化

```bash
cd /home/adama/wav2lip-project
source facefusion_env/bin/activate
```

### 2. 依存関係（インストール済み）

```bash
# 既にインストール済みの主要パッケージ
- numpy>=2.0.0
- onnxruntime-gpu>=1.19.2
- opencv-python>=4.10.0
- insightface
- gradio>=5.0.0
```

### 3. ディレクトリ構造

```
wav2lip-project/
├── facefusion_env/           # 専用仮想環境
├── facefusion/               # FaceFusionソースコード
├── models/facefusion/        # モデル格納（自動ダウンロード）
├── input/                    # 入力ファイル
│   ├── source_face.jpg       # ソース顔画像
│   └── target_video_3s.mp4   # ターゲット動画
└── output/facefusion/        # 出力ファイル
```

## 使用方法（推奨）

### 🚀 CLI順次処理版（最適化済み）

**Step 1: Face Swapper (顔交換)**
```bash
cd facefusion
source ../facefusion_env/bin/activate

# CUDA環境変数設定（重要）
export LD_LIBRARY_PATH="/home/adama/wav2lip-project/facefusion_env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/adama/wav2lip-project/facefusion_env/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/adama/wav2lip-project/facefusion_env/lib/python3.10/site-packages/nvidia/cublas/lib:/home/adama/wav2lip-project/facefusion_env/lib/python3.10/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH"

# Step 1: 顔交換処理
python facefusion.py headless-run \
  --source-paths ../input/source_face.jpg \
  --target-path ../input/target_video_3s.mp4 \
  --output-path ../output/facefusion/swap_result.mp4 \
  --execution-providers cuda \
  --face-detector-model retinaface \
  --face-detector-size 320x320 \
  --face-swapper-model inswapper_128_fp16 \
  --processors face_swapper \
  --execution-thread-count 2 \
  --video-memory-strategy tolerant \
  --log-level info
```

**Step 2: GFPGAN Enhancement (顔品質向上)**
```bash
# Step 2: GFPGAN顔品質向上
python facefusion.py headless-run \
  --source-paths ../output/facefusion/swap_result.mp4 \
  --target-path ../output/facefusion/swap_result.mp4 \
  --output-path ../output/facefusion/final_result.mp4 \
  --execution-providers cuda \
  --processors face_enhancer \
  --face-enhancer-model gfpgan_1.4 \
  --face-enhancer-blend 25 \
  --face-enhancer-weight 0.5 \
  --execution-thread-count 2 \
  --video-memory-strategy tolerant \
  --log-level info
```

**処理時間（実測値）**:
- Step 1 (Face Swap): ~30秒
- Step 2 (GFPGAN): ~58秒
- **合計: ~88秒** (GPU加速, RTX 3050 4GB)

### 一括処理スクリプト（オプション）

```bash
#!/bin/bash
# facefusion_batch.sh
cd /home/adama/wav2lip-project/facefusion
source ../facefusion_env/bin/activate

# CUDA環境変数設定
export LD_LIBRARY_PATH="/home/adama/wav2lip-project/facefusion_env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/adama/wav2lip-project/facefusion_env/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/adama/wav2lip-project/facefusion_env/lib/python3.10/site-packages/nvidia/cublas/lib:/home/adama/wav2lip-project/facefusion_env/lib/python3.10/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH"

echo "Face Swapper開始..."
python facefusion.py headless-run \
  --source-paths ../input/source_face.jpg \
  --target-path ../input/target_video_3s.mp4 \
  --output-path ../output/facefusion/swap_result.mp4 \
  --execution-providers cuda \
  --face-detector-model retinaface \
  --face-detector-size 320x320 \
  --face-swapper-model inswapper_128_fp16 \
  --processors face_swapper \
  --execution-thread-count 2 \
  --video-memory-strategy tolerant \
  --log-level info

echo "GFPGAN Enhancement開始..."
python facefusion.py headless-run \
  --source-paths ../output/facefusion/swap_result.mp4 \
  --target-path ../output/facefusion/swap_result.mp4 \
  --output-path ../output/facefusion/final_result.mp4 \
  --execution-providers cuda \
  --processors face_enhancer \
  --face-enhancer-model gfpgan_1.4 \
  --face-enhancer-blend 25 \
  --face-enhancer-weight 0.5 \
  --execution-thread-count 2 \
  --video-memory-strategy tolerant \
  --log-level info

echo "処理完了！出力ファイル: ../output/facefusion/final_result.mp4"
```

## 重要な設定パラメータ

### 最適化済み設定（推奨）
- `--face-detector-model retinaface`: 高精度顔検出
- `--face-detector-size 320x320`: バランス型解像度
- `--face-swapper-model inswapper_128_fp16`: 高速・高品質
- `--face-enhancer-model gfpgan_1.4`: 顔品質向上
- `--face-enhancer-blend 25`: 25%ブレンド（自然な仕上がり）
- `--execution-thread-count 2`: 2スレッド（4GB VRAM対応）
- `--video-memory-strategy tolerant`: メモリ寛容設定
- `--execution-providers cuda`: GPU加速必須

### パフォーマンス調整
- **高速化**: `--face-detector-size 160x160` (精度低下)
- **高品質**: `--face-enhancer-blend 50` (処理時間増加)
- **VRAM節約**: `--execution-thread-count 1` (速度低下)

## GPU環境セットアップ（重要）

### CUDA環境変数設定

GPU使用には必須の環境変数設定：

```bash
# 必須：CUDA環境変数設定
export LD_LIBRARY_PATH="/home/adama/wav2lip-project/facefusion_env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/adama/wav2lip-project/facefusion_env/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/adama/wav2lip-project/facefusion_env/lib/python3.10/site-packages/nvidia/cublas/lib:/home/adama/wav2lip-project/facefusion_env/lib/python3.10/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH"
```

### GPU動作確認

```bash
cd facefusion
source ../facefusion_env/bin/activate
# 上記環境変数設定後

# GPU対応確認
python -c "import onnxruntime as ort; print('CUDA available:', 'CUDAExecutionProvider' in ort.get_available_providers())"

# 結果: CUDA available: True であることを確認
```

## トラブルシューティング

### GPUエラー
```bash
# GPU利用不可の場合
--execution-providers cpu

# VRAM不足の場合
--video-memory-strategy tolerant
--execution-thread-count 1
```

### 環境変数エラー
```bash
# libcudnn.so.9 などのエラーが出た場合
echo $LD_LIBRARY_PATH  # 環境変数確認

# 上記のCUDA環境変数設定を再実行
export LD_LIBRARY_PATH="/home/adama/wav2lip-project/facefusion_env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/adama/wav2lip-project/facefusion_env/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/adama/wav2lip-project/facefusion_env/lib/python3.10/site-packages/nvidia/cublas/lib:/home/adama/wav2lip-project/facefusion_env/lib/python3.10/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH"
```

## 処理時間・品質

### 最適化済みパフォーマンス（RTX 3050 4GB）
- **Face Swap**: ~30秒
- **GFPGAN Enhancement**: ~58秒
- **合計**: ~88秒 (3秒動画)
- **出力品質**: 高品質・自然な仕上がり

### システム要件
- **VRAM**: 4GB以上（推奨）
- **RAM**: 8GB以上
- **GPU**: RTX 3050以上


## 注意事項

- 研究・個人利用目的でのみ使用
- 他人の同意なく顔交換を行わないこと
- 生成された動画の悪用を禁止

## 更新履歴

### v2.0 (2025-09-16)
- **CLI順次処理版完成**
  - Face Swap + GFPGAN 順次処理で最適化
  - GPU加速で~88秒で処理完了
  - CUDA環境変数設定で安定動作
  - サブプロセス版を廃止、CLI直接実行に一本化
  - READMEをシンプル化、必要最小限の情報のみ

---

**最終更新**: 2025-09-16
**GPU対応**: CUDA 12.6 + ONNX Runtime GPU 1.19.2
**推奨方法**: CLI順次処理（Face Swap → GFPGAN）