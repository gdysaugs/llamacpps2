#!/usr/bin/env python3
"""
FaceFusion Optimized Version
FP16 + RetinaFace + 顔領域クロップによる最適化版
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def main():
    print("=" * 60)
    print("FaceFusion Optimized (FP16 + RetinaFace + Face Crop)")
    print("=" * 60)

    # パス設定
    work_dir = Path("/home/adama/wav2lip-project")
    source_image = work_dir / "input/source_face.jpg"
    target_video = work_dir / "input/target_video_3s.mp4"
    output_video = work_dir / f"output/facefusion/optimized_{int(time.time())}.mp4"

    # ディレクトリ作成
    output_video.parent.mkdir(parents=True, exist_ok=True)

    # 環境変数設定
    env = os.environ.copy()

    # CUDA ライブラリパス設定
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

    # 最適化された FaceFusion コマンド
    cmd = [
        sys.executable,
        str(work_dir / "facefusion/facefusion.py"),
        "headless-run",

        # 入出力パス
        "--source-paths", str(source_image),
        "--target-path", str(target_video),
        "--output-path", str(output_video),

        # GPU実行
        "--execution-providers", "cuda",

        # ======== RetinaFace設定 ========
        "--face-detector-model", "retinaface",
        "--face-detector-size", "320x320",  # 低解像度で高速化
        "--face-detector-score", "0.6",     # 適度な閾値で誤検出を防ぐ

        # ======== FP16モデル ========
        "--face-swapper-model", "inswapper_128_fp16",  # メモリ効率&高速

        # ======== 顔領域クロップ処理 ========
        "--face-mask-types", "region",        # 顔領域のみ処理
        "--face-mask-blur", "0.3",           # 最小限のブラー
        "--face-mask-padding", "10", "10", "10", "10", # 適度なパディング

        # プロセッサ
        "--processors", "face_swapper",

        # ======== 出力最適化 ========
        "--output-video-encoder", "libx264",
        "--output-video-preset", "superfast",  # 高速エンコード
        "--output-video-quality", "20",        # バランスの良い品質

        # ======== メモリ最適化 ========
        "--system-memory-limit", "8",
        "--video-memory-strategy", "tolerant",  # メモリ使用を緩和

        # ======== 並列処理 ========
        "--execution-thread-count", "8",  # マルチスレッド処理

        # ログ設定
        "--log-level", "info"
    ]

    print("\n最適化設定:")
    print("-" * 40)
    print("✅ RetinaFace (320x320) - バランス型顔検出")
    print("✅ InSwapper 128 FP16 - 高速&メモリ効率")
    print("✅ 顔領域クロップ処理 - 処理範囲を顔だけに限定")
    print("✅ 8スレッド並列処理 - CPU/GPU並列化")
    print("✅ Tolerantメモリ戦略 - VRAM使用最適化")
    print("-" * 40)

    # 実行
    try:
        start_time = time.time()
        print(f"\n処理開始: {time.strftime('%H:%M:%S')}")
        print("-" * 40)

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(work_dir / "facefusion")
        )

        # リアルタイム出力とFPS計測
        frame_count = 0
        last_frame_count = 0
        fps_values = []

        for line in process.stdout:
            print(line, end='')

            # FPS情報を抽出
            if "Processing:" in line and "frame/s" in line:
                try:
                    # FPS値を抽出
                    if "frame/s," in line:
                        fps_part = line.split("frame/s,")[0].split()[-1]
                        fps = float(fps_part)
                        fps_values.append(fps)
                    frame_count += 1
                except:
                    pass

        # プロセス終了待機
        return_code = process.wait()
        elapsed_time = time.time() - start_time

        print("\n" + "=" * 60)

        if return_code == 0:
            print("✅ 処理完了！")
            print(f"処理時間: {elapsed_time:.2f}秒")

            # FPS統計
            if fps_values:
                avg_fps = sum(fps_values) / len(fps_values)
                max_fps = max(fps_values)
                min_fps = min(fps_values)
                print(f"平均FPS: {avg_fps:.2f}")
                print(f"最大FPS: {max_fps:.2f}")
                print(f"最小FPS: {min_fps:.2f}")
            elif frame_count > 0:
                fps = frame_count / elapsed_time
                print(f"推定FPS: {fps:.2f}")

            print(f"\n出力ファイル: {output_video}")

            # ファイルサイズ確認
            if output_video.exists():
                size_mb = output_video.stat().st_size / (1024 * 1024)
                print(f"ファイルサイズ: {size_mb:.2f} MB")

            # 改善度計算（ベースライン2.5 FPSと比較）
            if fps_values:
                improvement = (avg_fps / 2.5 - 1) * 100
                if improvement > 0:
                    print(f"\n🚀 速度改善: +{improvement:.0f}%")

        else:
            print(f"❌ エラー: 終了コード {return_code}")

        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  処理を中断しました")
        process.terminate()
        process.wait()

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