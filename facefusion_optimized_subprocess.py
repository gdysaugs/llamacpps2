#!/usr/bin/env python3
"""
FaceFusion Optimized Subprocess Version
RetinaFace + FP16 + CUDA環境変数設定による最適化版
"""

import subprocess
import sys
import os
import time
import psutil
from pathlib import Path

def setup_cuda_environment():
    """CUDA環境変数を設定"""
    work_dir = Path("/home/adama/wav2lip-project")
    venv_lib = work_dir / "facefusion_env/lib/python3.10/site-packages"

    cuda_paths = [
        venv_lib / "nvidia/cuda_runtime/lib",
        venv_lib / "nvidia/cudnn/lib",
        venv_lib / "nvidia/cublas/lib",
        venv_lib / "nvidia/cufft/lib"
    ]

    ld_library_path = ":".join(str(p) for p in cuda_paths)
    current_path = os.environ.get("LD_LIBRARY_PATH", "")
    if current_path:
        ld_library_path = f"{ld_library_path}:{current_path}"

    return ld_library_path

def monitor_process(process, timeout=300):
    """プロセス監視とメモリ使用量チェック"""
    start_time = time.time()
    max_memory = 0

    while process.poll() is None:
        try:
            # メモリ使用量監視
            proc_info = psutil.Process(process.pid)
            memory_mb = proc_info.memory_info().rss / 1024 / 1024
            max_memory = max(max_memory, memory_mb)

            # タイムアウトチェック
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"\n⚠️  タイムアウト ({timeout}秒) - プロセスを終了します")
                process.terminate()
                process.wait()
                return False, max_memory, elapsed

        except psutil.NoSuchProcess:
            break

        time.sleep(5)

    elapsed = time.time() - start_time
    return process.returncode == 0, max_memory, elapsed

def main():
    print("=" * 60)
    print("FaceFusion Optimized Subprocess")
    print("RetinaFace + InSwapper FP16 + CUDA")
    print("=" * 60)

    # パス設定
    work_dir = Path("/home/adama/wav2lip-project")
    source_image = work_dir / "input/source_face.jpg"
    target_video = work_dir / "input/target_video_3s.mp4"
    output_video = work_dir / f"output/facefusion/optimized_subprocess_{int(time.time())}.mp4"

    # ディレクトリ作成
    output_video.parent.mkdir(parents=True, exist_ok=True)

    # 入力ファイル確認
    if not source_image.exists():
        print(f"❌ ソース画像が見つかりません: {source_image}")
        return

    if not target_video.exists():
        print(f"❌ ターゲット動画が見つかりません: {target_video}")
        return

    print(f"\n入力ファイル:")
    print(f"  ソース: {source_image}")
    print(f"  ターゲット: {target_video}")
    print(f"  出力: {output_video}")

    # CUDA環境設定
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = setup_cuda_environment()

    # 仮想環境パス
    venv_python = work_dir / "facefusion_env/bin/python"
    facefusion_script = work_dir / "facefusion/facefusion.py"

    # 順次処理: Face Swapper → GFPGAN
    # Step 1: Face Swapper
    temp_video = work_dir / f"output/facefusion/temp_{int(time.time())}.mp4"

    cmd1 = [
        str(venv_python),
        str(facefusion_script),
        "headless-run",
        "--source-paths", str(source_image),
        "--target-path", str(target_video),
        "--output-path", str(temp_video),
        "--execution-providers", "cuda",
        "--face-detector-model", "retinaface",
        "--face-detector-size", "320x320",
        "--face-swapper-model", "inswapper_128_fp16",
        "--processors", "face_swapper",
        "--execution-thread-count", "2",
        "--video-memory-strategy", "tolerant",
        "--log-level", "info"
    ]

    # Step 2: GFPGAN Enhancement
    cmd2 = [
        str(venv_python),
        str(facefusion_script),
        "headless-run",
        "--source-paths", str(temp_video),
        "--target-path", str(temp_video),
        "--output-path", str(output_video),
        "--execution-providers", "cuda",
        "--processors", "face_enhancer",
        "--face-enhancer-model", "gfpgan_1.4",
        "--face-enhancer-blend", "25",
        "--face-enhancer-weight", "0.5",
        "--execution-thread-count", "2",
        "--video-memory-strategy", "tolerant",
        "--log-level", "info"
    ]

    print(f"\n最適化設定:")
    print("-" * 40)
    print("✅ 順次処理: Face Swap → GFPGAN (OOM回避)")
    print("✅ Step 1: InSwapper 128 FP16")
    print("✅ Step 2: GFPGAN 1.4 (25%ブレンド)")
    print("✅ CUDA GPU実行 - 環境変数自動設定")
    print("✅ 出力最適化 - DEVNULL, 5秒間隔監視")
    print("✅ 予想時間: 約80-90秒 (3.4倍高速化)")
    print("-" * 40)

    # サブプロセス実行
    try:
        print(f"\n処理開始: {time.strftime('%H:%M:%S')}")
        print("-" * 40)

        # メモリ使用量（開始前）
        initial_memory = psutil.virtual_memory().percent

        # Step 1: Face Swapper実行
        print(f"\n[Step 1/2] Face Swapper実行中...")
        process1 = subprocess.Popen(
            cmd1,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(work_dir / "facefusion")
        )
        success1, max_memory1, elapsed_time1 = monitor_process(process1, timeout=300)

        if not success1:
            print("❌ Face Swapper失敗")
            return

        print(f"✅ Step 1完了: {elapsed_time1:.2f}秒")

        # Step 2: GFPGAN Enhancement実行
        print(f"\n[Step 2/2] GFPGAN Enhancement実行中...")
        process2 = subprocess.Popen(
            cmd2,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(work_dir / "facefusion")
        )
        success2, max_memory2, elapsed_time2 = monitor_process(process2, timeout=300)

        if not success2:
            print("❌ GFPGAN Enhancement失敗")
            return

        print(f"✅ Step 2完了: {elapsed_time2:.2f}秒")

        # 全体の結果
        success = success1 and success2
        max_memory = max(max_memory1, max_memory2)
        elapsed_time = elapsed_time1 + elapsed_time2

        print("\n" + "=" * 60)

        if success:
            print("✅ 全処理完了！")
            print(f"Step 1 (Face Swap): {elapsed_time1:.2f}秒")
            print(f"Step 2 (GFPGAN): {elapsed_time2:.2f}秒")
            print(f"合計処理時間: {elapsed_time:.2f}秒")
            print(f"最大メモリ使用量: {max_memory:.0f} MB")

            # FPS計算（90フレーム想定）
            fps = 90 / elapsed_time
            print(f"平均FPS: {fps:.2f}")

            # 出力ファイル確認
            if output_video.exists():
                size_mb = output_video.stat().st_size / (1024 * 1024)
                print(f"出力ファイルサイズ: {size_mb:.2f} MB")
                print(f"出力パス: {output_video}")

                # 改善度表示（同時処理との比較）
                baseline_time = 280  # 同時処理の実測値
                improvement = (baseline_time / elapsed_time - 1) * 100
                print(f"\n🚀 速度改善: +{improvement:.0f}% (同時処理比)")

                # 一時ファイル削除
                if temp_video.exists():
                    temp_video.unlink()
                    print("一時ファイル削除済み")
            else:
                print("⚠️  出力ファイルが見つかりません")

        else:
            print("❌ 処理失敗またはタイムアウト")
            print(f"経過時間: {elapsed_time:.2f}秒")

        # メモリクリーンアップ
        final_memory = psutil.virtual_memory().percent
        print(f"\nメモリ使用量: {initial_memory:.1f}% → {final_memory:.1f}%")

        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  処理を中断しました")
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()

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