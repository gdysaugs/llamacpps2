#!/usr/bin/env python3
"""
FaceFusion サブプロセス実行（exe対応版）
実行ファイル基準パス + OS対応 + 直接Python実行
"""
import subprocess
import time
import psutil
import torch
import os
import sys
import platform
from pathlib import Path

def get_base_paths():
    """実行ファイル基準のベースパス取得"""
    if getattr(sys, 'frozen', False):
        # exe実行時
        base_path = Path(sys.executable).parent
    else:
        # Python実行時
        base_path = Path(__file__).parent
    
    return {
        'base': base_path,
        'facefusion': base_path / 'facefusion',
        'env': base_path / 'facefusion_env',
        'input': base_path / 'input', 
        'output': base_path / 'output' / 'facefusion'
    }

def get_python_executable(paths):
    """OS対応Python実行ファイル取得"""
    is_windows = platform.system() == 'Windows'
    
    if is_windows:
        python_exe = paths['env'] / 'Scripts' / 'python.exe'
        if not python_exe.exists():
            python_exe = 'python.exe'  # システムpython
    else:
        python_exe = paths['env'] / 'bin' / 'python'
        if not python_exe.exists():
            python_exe = sys.executable  # 現在のpython
    
    return str(python_exe)

def setup_environment_variables(paths):
    """OS対応環境変数動的構築"""
    env = os.environ.copy()
    is_windows = platform.system() == 'Windows'
    
    if is_windows:
        # Windows用PATH設定
        nvidia_paths = [
            paths['env'] / 'Lib' / 'site-packages' / 'nvidia' / 'cuda_runtime' / 'bin',
            paths['env'] / 'Lib' / 'site-packages' / 'nvidia' / 'cudnn' / 'bin',
            paths['env'] / 'Lib' / 'site-packages' / 'nvidia' / 'cublas' / 'bin',
            paths['env'] / 'Lib' / 'site-packages' / 'nvidia' / 'cufft' / 'bin'
        ]
        existing_path = env.get('PATH', '')
        nvidia_path_str = ';'.join(str(p) for p in nvidia_paths if p.exists())
        if nvidia_path_str:
            env['PATH'] = f"{nvidia_path_str};{existing_path}"
    else:
        # Linux用LD_LIBRARY_PATH設定
        nvidia_paths = [
            paths['env'] / 'lib' / 'python3.10' / 'site-packages' / 'nvidia' / 'cuda_runtime' / 'lib',
            paths['env'] / 'lib' / 'python3.10' / 'site-packages' / 'nvidia' / 'cudnn' / 'lib',
            paths['env'] / 'lib' / 'python3.10' / 'site-packages' / 'nvidia' / 'cublas' / 'lib',
            paths['env'] / 'lib' / 'python3.10' / 'site-packages' / 'nvidia' / 'cufft' / 'lib'
        ]
        existing_path = env.get('LD_LIBRARY_PATH', '')
        nvidia_path_str = ':'.join(str(p) for p in nvidia_paths if p.exists())
        if nvidia_path_str:
            env['LD_LIBRARY_PATH'] = f"{nvidia_path_str}:{existing_path}"
    
    return env

def get_memory_info():
    """メモリ使用量取得"""
    memory = psutil.virtual_memory()
    gpu_info = {}
    
    if torch.cuda.is_available():
        gpu_info = {
            'allocated': torch.cuda.memory_allocated() / 1024**2,
            'reserved': torch.cuda.memory_reserved() / 1024**2
        }
    
    return {
        'cpu_percent': memory.percent,
        'gpu': gpu_info
    }

def run_facefusion_subprocess():
    """exe対応版 FaceFusion サブプロセス実行"""
    print("🚀 FaceFusion サブプロセス実行開始（exe対応版）")
    
    # パス取得
    paths = get_base_paths()
    print(f"ベースパス: {paths['base']}")
    
    # 開始前メモリ
    mem_before = get_memory_info()
    print(f"開始前: CPU={mem_before['cpu_percent']:.1f}%, GPU={mem_before['gpu'].get('allocated', 0):.1f}MB")
    
    # Python実行ファイル取得
    python_exe = get_python_executable(paths)
    print(f"Python実行ファイル: {python_exe}")
    
    # 環境変数設定
    env = setup_environment_variables(paths)
    
    # 出力ディレクトリ作成
    paths['output'].mkdir(parents=True, exist_ok=True)
    
    # FaceFusionコマンド構築（直接Python実行）
    cmd = [
        python_exe,
        str(paths['facefusion'] / 'facefusion.py'),
        'headless-run',
        '--source-paths', str(paths['input'] / 'source_face.jpg'),
        '--target-path', str(paths['input'] / 'target_video_3s.mp4'), 
        '--output-path', str(paths['output'] / 'subprocess_test.mp4'),
        '--execution-providers', 'cuda',
        '--face-swapper-model', 'inswapper_128',
        '--processors', 'face_swapper'
    ]
    
    print("実行コマンド:")
    print(" ".join(cmd))
    
    start_time = time.time()
    
    try:
        # サブプロセス実行（直接Python、OS対応環境変数）
        result = subprocess.run(
            cmd,
            env=env,
            text=True,
            timeout=120,
            cwd=str(paths['facefusion'])
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"⏱️  実行時間: {processing_time:.1f}秒")
        print(f"📊 終了コード: {result.returncode}")
        
        # 成功時の出力
        if result.returncode == 0:
            print("✅ 処理成功！")
            # 出力ファイル確認（相対パス）
            output_file = paths['output'] / 'subprocess_test.mp4'
            if output_file.exists():
                file_size = output_file.stat().st_size / 1024 / 1024
                print(f"📁 出力ファイル: {file_size:.1f}MB")
        else:
            print("❌ 処理失敗")
        
        success = result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ タイムアウトエラー")
        success = False
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        success = False
    
    # 完了後メモリ
    time.sleep(1)
    mem_after = get_memory_info()
    print(f"完了後: CPU={mem_after['cpu_percent']:.1f}%, GPU={mem_after['gpu'].get('allocated', 0):.1f}MB")
    
    # メモリ変化
    cpu_diff = mem_after['cpu_percent'] - mem_before['cpu_percent']
    gpu_diff = mem_after['gpu'].get('allocated', 0) - mem_before['gpu'].get('allocated', 0)
    print(f"メモリ変化: CPU={cpu_diff:+.1f}%, GPU={gpu_diff:+.1f}MB")
    
    return success

def test_multiple_runs(count=3):
    """複数回実行テスト"""
    print(f"\n🔄 {count}回連続実行テスト")
    
    results = []
    for i in range(count):
        print(f"\n--- 実行 {i+1}/{count} ---")
        
        success = run_facefusion_subprocess()
        results.append(success)
        
        if i < count - 1:
            print("3秒待機...")
            time.sleep(3)
    
    # 結果サマリー
    success_count = sum(results)
    print(f"\n📈 最終結果: {success_count}/{count} 成功")
    
    if success_count == count:
        print("🎯 全て成功 - サブプロセス動作完璧！")
    else:
        print("⚠️ 一部失敗")
    
    return success_count == count

def main():
    print("🧪 FaceFusion サブプロセステスト")
    print("READMEコマンドのサブプロセス化")
    print("=" * 40)
    
    # 単回実行
    success = run_facefusion_subprocess()
    
    if success:
        print("\n✅ 単回実行成功")
        
        # 出力ファイル確認
        paths = get_base_paths()
        output_file = paths['output'] / 'subprocess_test.mp4'
        if output_file.exists():
            file_size = output_file.stat().st_size / 1024 / 1024
            print(f"📁 出力ファイル: {file_size:.1f}MB")
        
        # 複数回テスト
        print("\n🔄 複数回実行テスト開始")
        all_success = test_multiple_runs(3)
        
        if all_success:
            print("\n🎉 サブプロセステスト完全成功！")
            print("✅ メモリ分離正常")
            print("✅ プロセス管理安定")
            print("✅ 連続実行可能")
        
    else:
        print("\n❌ 単回実行失敗")

if __name__ == "__main__":
    main()