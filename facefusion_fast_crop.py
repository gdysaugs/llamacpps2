#!/usr/bin/env python3
"""
FaceFusion Fast Crop Version
顔領域のみをクロップして処理することで高速化
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def main():
    print("=" * 60)
    print("FaceFusion Fast Crop Mode")
    print("=" * 60)

    # パス設定
    work_dir = Path("/home/adama/wav2lip-project")
    source_image = work_dir / "input/source_face.jpg"
    target_video = work_dir / "input/target_video_3s.mp4"
    output_video = work_dir / f"output/facefusion/fast_crop_{int(time.time())}.mp4"

    # 環境変数設定
    env = os.environ.copy()

    # CUDA ライブラリパス
    venv_lib = work_dir / "facefusion_env/lib/python3.10/site-packages"
    cuda_paths = [
        venv_lib / "nvidia/cuda_runtime/lib",
        venv_lib / "nvidia/cudnn/lib",
        venv_lib / "nvidia/cublas/lib",
        venv_lib / "nvidia/cufft/lib"
    ]

    ld_library_path = ":".join(str(p) for p in cuda_paths)
    if "LD_LIBRARY_PATH" in env:
        ld_library_path = f"{ld_library_path}:{env['LD_LIBRARY_PATH']}"
    env["LD_LIBRARY_PATH"] = ld_library_path

    # 高速化設定
    cmd = [
        sys.executable,
        str(work_dir / "facefusion/facefusion.py"),
        "headless-run",
        "--source-paths", str(source_image),
        "--target-path", str(target_video),
        "--output-path", str(output_video),

        # 実行プロバイダ
        "--execution-providers", "cuda",

        # RetinaFaceに変更（バランス良い）
        "--face-detector-model", "retinaface",
        "--face-detector-size", "320x320",  # 低解像度で高速化
        "--face-detector-score", "0.7",     # 閾値を上げて検出数削減

        # FP16モデル使用（メモリ効率&高速）
        "--face-swapper-model", "inswapper_128_fp16",

        # プロセッサ
        "--processors", "face_swapper",

        # 顔領域のクロップ処理を有効化
        "--face-mask-types", "region",      # 顔領域のみ処理
        "--face-mask-blur", "0.0",          # ブラー無効で高速化
        "--face-mask-padding", "5,5,5,5",   # 最小パディング

        # 出力設定
        "--output-video-encoder", "libx264",
        "--output-video-preset", "ultrafast",  # 最速エンコード
        "--output-video-quality", "23",        # 品質を少し下げて高速化
        "--output-video-resolution", "640x480", # 解像度下げる

        # メモリ設定
        "--system-memory-limit", "8",
        "--video-memory-strategy", "tolerant",

        # スレッド設定
        "--execution-thread-count", "8",

        # ログレベル
        "--log-level", "info"
    ]

    print("\n設定:")
    print("-" * 40)
    print("✅ RetinaFace (320x320) - バランス型顔検出")
    print("✅ InSwapper 128 FP16 - 高速&メモリ効率")
    print("✅ 顔領域クロップ処理 - 処理範囲限定")
    print("✅ 解像度 640x480 - 低解像度で高速化")
    print("✅ Ultrafast preset - 最速エンコード")
    print("-" * 40)

    # 実行
    try:
        start_time = time.time()

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(work_dir / "facefusion")
        )

        # リアルタイム出力
        frame_count = 0
        for line in process.stdout:
            print(line, end='')
            if "Processing:" in line and "frame/s" in line:
                frame_count += 1

        # 終了待機
        return_code = process.wait()
        elapsed_time = time.time() - start_time

        if return_code == 0:
            print("\n" + "=" * 60)
            print("✅ 処理完了！")
            print(f"処理時間: {elapsed_time:.2f}秒")

            if frame_count > 0:
                fps = frame_count / elapsed_time
                print(f"平均FPS: {fps:.2f}")

            print(f"出力: {output_video}")

            if output_video.exists():
                size_mb = output_video.stat().st_size / (1024 * 1024)
                print(f"ファイルサイズ: {size_mb:.2f} MB")

            print("=" * 60)
        else:
            print(f"\n❌ エラー: 終了コード {return_code}")

    except Exception as e:
        print(f"\n❌ 実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 仮想環境チェック
    if not sys.prefix.endswith("facefusion_env"):
        print("⚠️  仮想環境を有効化してください:")
        print("source /home/adama/wav2lip-project/facefusion_env/bin/activate")
        sys.exit(1)

    main()