#!/bin/bash
# FaceFusion 高速化テストスクリプト

cd /home/adama/wav2lip-project/facefusion
source ../facefusion_env/bin/activate
export LD_LIBRARY_PATH="../facefusion_env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:../facefusion_env/lib/python3.10/site-packages/nvidia/cudnn/lib:../facefusion_env/lib/python3.10/site-packages/nvidia/cublas/lib:../facefusion_env/lib/python3.10/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH"

echo "=== FaceFusion 高速化テスト ==="

# テスト1: 現在の設定（ベースライン）
echo "テスト1: 現在設定（ベースライン）"
time python facefusion.py headless-run \
  --source-paths ../input/source_face.jpg \
  --target-path ../input/target_video_3s.mp4 \
  --output-path ../output/facefusion/test_baseline.mp4 \
  --execution-providers cuda \
  --face-swapper-model inswapper_128 \
  --processors face_swapper

# テスト2: FP16 + 最適化設定
echo -e "\nテスト2: FP16 + 最適化設定"
time python facefusion.py headless-run \
  --source-paths ../input/source_face.jpg \
  --target-path ../input/target_video_3s.mp4 \
  --output-path ../output/facefusion/test_fp16_optimized.mp4 \
  --execution-providers cuda \
  --face-swapper-model inswapper_128_fp16 \
  --processors face_swapper \
  --execution-thread-count 8 \
  --video-memory-strategy tolerant \
  --system-memory-limit 8 \
  --output-video-encoder h264_nvenc \
  --output-video-preset ultrafast \
  --output-video-quality 23

# テスト3: 超高速設定
echo -e "\nテスト3: 超高速設定"
time python facefusion.py headless-run \
  --source-paths ../input/source_face.jpg \
  --target-path ../input/target_video_3s.mp4 \
  --output-path ../output/facefusion/test_ultrafast.mp4 \
  --execution-providers cuda \
  --face-swapper-model inswapper_128_fp16 \
  --face-detector-model yolo_face \
  --processors face_swapper \
  --execution-thread-count 8 \
  --video-memory-strategy tolerant \
  --system-memory-limit 8 \
  --output-video-encoder h264_nvenc \
  --output-video-preset ultrafast \
  --output-video-quality 25

echo -e "\n=== 完了 ==="
ls -lah ../output/facefusion/test_*.mp4