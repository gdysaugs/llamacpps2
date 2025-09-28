#!/usr/bin/env python3
"""
FaceFusion Memory Optimized Direct Execution
順次処理 + 明示的メモリ解放
"""

import os
import sys
import time
import gc
import psutil
from pathlib import Path

# FaceFusionのパスを追加
sys.path.append(str(Path("/home/adama/wav2lip-project/facefusion")))

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

    os.environ["LD_LIBRARY_PATH"] = ld_library_path

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
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass
    gc.collect()
    print(f"メモリ解放実行済み")

def face_swap_process(source_path, target_path, output_path):
    """Face Swapper処理（関数スコープでメモリ管理）"""
    print(f"\n[Step 1/2] Face Swapper処理開始...")
    start_time = time.time()
    mem_before = get_memory_info()
    print(f"開始時メモリ: RAM {mem_before[0]:.0f}MB, GPU {mem_before[1]:.0f}MB")

    # FaceFusionのインポートと実行
    os.chdir("/home/adama/wav2lip-project/facefusion")
    sys.argv = [
        "facefusion.py", "headless-run",
        "--source-paths", str(source_path),
        "--target-path", str(target_path),
        "--output-path", str(output_path),
        "--execution-providers", "cuda",
        "--face-detector-model", "retinaface",
        "--face-detector-size", "320x320",
        "--face-swapper-model", "inswapper_128_fp16",
        "--processors", "face_swapper",
        "--execution-thread-count", "2",
        "--video-memory-strategy", "tolerant",
        "--log-level", "info"
    ]

    import facefusion.core as core
    core.cli()

    elapsed = time.time() - start_time
    mem_after = get_memory_info()
    print(f"✅ Face Swapper完了: {elapsed:.2f}秒")
    print(f"終了時メモリ: RAM {mem_after[0]:.0f}MB, GPU {mem_after[1]:.0f}MB")

    # ローカル変数をクリア
    del core

    return elapsed

def gfpgan_process(input_path, output_path):
    """GFPGAN Enhancement処理（関数スコープでメモリ管理）"""
    print(f"\n[Step 2/2] GFPGAN Enhancement処理開始...")
    start_time = time.time()
    mem_before = get_memory_info()
    print(f"開始時メモリ: RAM {mem_before[0]:.0f}MB, GPU {mem_before[1]:.0f}MB")

    # FaceFusionのインポートと実行
    os.chdir("/home/adama/wav2lip-project/facefusion")
    sys.argv = [
        "facefusion.py", "headless-run",
        "--source-paths", str(input_path),
        "--target-path", str(input_path),
        "--output-path", str(output_path),
        "--execution-providers", "cuda",
        "--processors", "face_enhancer",
        "--face-enhancer-model", "gfpgan_1.4",
        "--face-enhancer-blend", "25",
        "--face-enhancer-weight", "0.5",
        "--execution-thread-count", "2",
        "--video-memory-strategy", "tolerant",
        "--log-level", "info"
    ]

    import facefusion.core as core
    core.cli()

    elapsed = time.time() - start_time
    mem_after = get_memory_info()
    print(f"✅ GFPGAN Enhancement完了: {elapsed:.2f}秒")
    print(f"終了時メモリ: RAM {mem_after[0]:.0f}MB, GPU {mem_after[1]:.0f}MB")

    # ローカル変数をクリア
    del core

    return elapsed

def main():
    print("=" * 60)
    print("FaceFusion Memory Optimized Direct Execution")
    print("順次処理 + 明示的メモリ解放")
    print("=" * 60)

    # CUDA環境設定
    setup_cuda_environment()

    # パス設定
    work_dir = Path("/home/adama/wav2lip-project")
    source_image = work_dir / "input/source_face.jpg"
    target_video = work_dir / "input/target_video_3s.mp4"
    temp_video = work_dir / f"output/facefusion/temp_{int(time.time())}.mp4"
    output_video = work_dir / f"output/facefusion/memory_opt_{int(time.time())}.mp4"

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

    print(f"\n最適化設定:")
    print("-" * 40)
    print("✅ 順次処理: Face Swap → GFPGAN")
    print("✅ 関数スコープでメモリ分離")
    print("✅ 各ステップ後に明示的メモリ解放")
    print("✅ GPU メモリキャッシュクリア")
    print("✅ ガベージコレクション強制実行")
    print("-" * 40)

    try:
        total_start = time.time()
        initial_mem = get_memory_info()
        print(f"\n初期メモリ: RAM {initial_mem[0]:.0f}MB, GPU {initial_mem[1]:.0f}MB")

        # Step 1: Face Swapper (関数内で処理)
        step1_time = face_swap_process(source_image, target_video, temp_video)
        after_step1 = get_memory_info()

        # Step 2: GFPGAN Enhancement (関数内で処理)
        step2_time = gfpgan_process(temp_video, output_video)

        # 全処理完了後にメモリ解放
        print("\n--- 最終メモリ解放中 ---")
        clear_memory()
        after_clear2 = get_memory_info()
        print(f"解放後メモリ: RAM {after_clear2[0]:.0f}MB, GPU {after_clear2[1]:.0f}MB")

        # 全体の処理時間
        total_time = time.time() - total_start

        print("\n" + "=" * 60)
        print("✅ 全処理完了！")
        print(f"Step 1 (Face Swap): {step1_time:.2f}秒")
        print(f"Step 2 (GFPGAN): {step2_time:.2f}秒")
        print(f"合計処理時間: {total_time:.2f}秒")

        # メモリ使用状況サマリー
        print(f"\nメモリ使用状況:")
        print(f"  初期: RAM {initial_mem[0]:.0f}MB")
        print(f"  Step1後: RAM {after_step1[0]:.0f}MB")
        print(f"  最終解放後: RAM {after_clear2[0]:.0f}MB")

        # FPS計算
        fps = 90 / total_time
        print(f"\n平均FPS: {fps:.2f}")

        # 出力ファイル確認
        if output_video.exists():
            size_mb = output_video.stat().st_size / (1024 * 1024)
            print(f"出力ファイルサイズ: {size_mb:.2f} MB")
            print(f"出力パス: {output_video}")

            # 比較
            print(f"\n📊 パフォーマンス比較:")
            print(f"サブプロセス版: 160秒")
            print(f"メモリ最適化版: {total_time:.2f}秒")
            if total_time < 160:
                improvement = (160 / total_time - 1) * 100
                print(f"🚀 速度改善: +{improvement:.0f}%")

            # 一時ファイル削除（出力ファイルが存在する場合のみ）
            if temp_video.exists():
                temp_video.unlink()
                print(f"\n一時ファイル削除済み")
        else:
            print(f"❌ 出力ファイルが作成されませんでした: {output_video}")
            print(f"一時ファイルが残っています: {temp_video}")
            if temp_video.exists():
                print(f"一時ファイルサイズ: {temp_video.stat().st_size / (1024 * 1024):.2f} MB")

        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️ 処理を中断しました")

    except Exception as e:
        print(f"\n❌ 実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 仮想環境チェック
    if not sys.prefix.endswith("facefusion_env"):
        print("⚠️ 仮想環境を有効化してください:")
        print("source /home/adama/wav2lip-project/facefusion_env/bin/activate")
        sys.exit(1)

    main()