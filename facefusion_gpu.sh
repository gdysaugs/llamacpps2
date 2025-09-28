#!/bin/bash
# FaceFusion GPU環境設定スクリプト

export LD_LIBRARY_PATH="../facefusion_env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:../facefusion_env/lib/python3.10/site-packages/nvidia/cudnn/lib:../facefusion_env/lib/python3.10/site-packages/nvidia/cublas/lib:../facefusion_env/lib/python3.10/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH"

cd facefusion
source ../facefusion_env/bin/activate

echo "FaceFusion GPU環境が設定されました。"
echo "使用例："
echo "python facefusion.py headless-run --source-paths ../input/source_face.jpg --target-path ../input/target_video_3s.mp4 --output-path ../output/facefusion/result.mp4 --execution-providers cuda --face-swapper-model inswapper_128 --processors face_swapper"