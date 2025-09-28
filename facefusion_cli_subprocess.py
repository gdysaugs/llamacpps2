#!/usr/bin/env python3
"""
FaceFusion CLI Subprocess Version
成功したCLI順次処理（Face Swap → GFPGAN）をサブプロセスで実行
"""

import subprocess
import sys
import os
import time
import psutil
import gc
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

def get_memory_info():
    """現在のメモリ使用状況を取得"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    gpu_memory = get_gpu_memory()
    return memory_mb, gpu_memory

def get_gpu_memory():
    """GPU メモリ使用量を取得"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except:
        pass
    return 0

def clear_memory():
    """メモリを明示的に解放"""
    print("\n--- メモリ解放中 ---")

    # Python ガベージコレクション
    gc.collect()

    # GPU メモリクリア
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("GPU メモリキャッシュをクリア")
    except ImportError:
        pass
    except Exception as e:
        print(f"GPU メモリクリア中にエラー: {e}")

    # 再度ガベージコレクション
    gc.collect()

    print("メモリ解放完了")

def monitor_process(process, step_name, timeout=300):
    """プロセス監視とメモリ使用量チェック"""
    start_time = time.time()
    max_memory = 0

    print(f"[{step_name}] 実行中...")

    while process.poll() is None:
        try:
            # メモリ使用量監視
            proc_info = psutil.Process(process.pid)
            memory_mb = proc_info.memory_info().rss / 1024 / 1024
            max_memory = max(max_memory, memory_mb)

            # タイムアウトチェック
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"\n⚠️ タイムアウト ({timeout}秒) - プロセスを終了します")
                process.terminate()
                process.wait()
                return False, max_memory, elapsed

        except psutil.NoSuchProcess:
            break

        time.sleep(2)  # 2秒間隔で監視

    elapsed = time.time() - start_time
    success = process.returncode == 0

    if success:
        print(f"✅ [{step_name}] 完了: {elapsed:.2f}秒")
    else:
        print(f"❌ [{step_name}] 失敗: {elapsed:.2f}秒")

    return success, max_memory, elapsed

def main():
    print("=" * 60)
    print("FaceFusion CLI Subprocess")
    print("CLI順次処理（Face Swap → GFPGAN）をサブプロセスで実行")
    print("=" * 60)

    # パス設定
    work_dir = Path("/home/adama/wav2lip-project")
    source_image = work_dir / "input/source_face.jpg"
    target_video = work_dir / "input/target_video_3s.mp4"
    temp_video = work_dir / f"output/facefusion/temp_{int(time.time())}.mp4"
    output_video = work_dir / f"output/facefusion/cli_subprocess_{int(time.time())}.mp4"

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

    # Step 1: Face Swapper
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
    print("✅ CLI順次処理: Face Swap → GFPGAN")
    print("✅ RetinaFace (320x320) + InSwapper 128 FP16")
    print("✅ GFPGAN 1.4 (25%ブレンド)")
    print("✅ CUDA GPU実行 - 環境変数自動設定")
    print("✅ 予想時間: 約88秒")
    print("-" * 40)

    try:
        print(f"\n処理開始: {time.strftime('%H:%M:%S')}")
        print("-" * 40)
        total_start = time.time()

        # 初期メモリ状況
        initial_mem = get_memory_info()
        print(f"初期メモリ: RAM {initial_mem[0]:.0f}MB, GPU {initial_mem[1]:.0f}MB")

        # Step 1: Face Swapper実行
        process1 = subprocess.Popen(
            cmd1,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(work_dir / "facefusion")
        )
        success1, max_memory1, elapsed_time1 = monitor_process(process1, "Face Swap", timeout=300)

        if not success1:
            print("❌ Face Swapper失敗")
            return

        # Step 2: GFPGAN Enhancement実行
        process2 = subprocess.Popen(
            cmd2,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(work_dir / "facefusion")
        )
        success2, max_memory2, elapsed_time2 = monitor_process(process2, "GFPGAN Enhancement", timeout=300)

        if not success2:
            print("❌ GFPGAN Enhancement失敗")
            return

        # 全体の結果
        total_elapsed = time.time() - total_start
        max_memory = max(max_memory1, max_memory2)

        # 処理完了後のメモリ解放
        clear_memory()

        # メモリ解放後の状況確認
        final_mem = get_memory_info()

        print("\n" + "=" * 60)
        print("✅ 全処理完了！")
        print(f"Step 1 (Face Swap): {elapsed_time1:.2f}秒")
        print(f"Step 2 (GFPGAN): {elapsed_time2:.2f}秒")
        print(f"合計処理時間: {total_elapsed:.2f}秒")
        print(f"最大メモリ使用量: {max_memory:.0f} MB")

        # メモリ使用状況サマリー
        print(f"\nメモリ使用状況:")
        print(f"  初期: RAM {initial_mem[0]:.0f}MB, GPU {initial_mem[1]:.0f}MB")
        print(f"  解放後: RAM {final_mem[0]:.0f}MB, GPU {final_mem[1]:.0f}MB")

        # 出力ファイル確認
        if output_video.exists():
            size_mb = output_video.stat().st_size / (1024 * 1024)
            print(f"出力ファイルサイズ: {size_mb:.2f} MB")
            print(f"出力パス: {output_video}")

            # 一時ファイル削除
            if temp_video.exists():
                temp_video.unlink()
                print("一時ファイル削除済み")

            # パフォーマンス情報
            fps = 90 / total_elapsed
            print(f"平均FPS: {fps:.2f}")
            print(f"🚀 CLI順次処理をサブプロセス化に成功")
        else:
            print("❌ 出力ファイルが作成されませんでした")

        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️ 処理を中断しました")
        try:
            if 'process1' in locals():
                process1.terminate()
                process1.wait(timeout=5)
            if 'process2' in locals():
                process2.terminate()
                process2.wait(timeout=5)
        except:
            pass

        # 中断時もメモリ解放を実行
        try:
            clear_memory()
        except:
            pass

    except Exception as e:
        print(f"\n❌ 実行エラー: {e}")
        import traceback
        traceback.print_exc()

        # エラー時もメモリ解放を実行
        try:
            clear_memory()
        except:
            pass

if __name__ == "__main__":
    # 仮想環境チェック
    if not sys.prefix.endswith("facefusion_env"):
        print("⚠️ 仮想環境を有効化してください:")
        print("source /home/adama/wav2lip-project/facefusion_env/bin/activate")
        sys.exit(1)

    main()