#!/usr/bin/env python3
"""
FaceFusion サブプロセステスト
メモリ分離とプロセス管理の安全性確認
"""
import subprocess
import time
import psutil
import torch
import os
import signal
import sys

def get_memory_info():
    """現在のメモリ使用量を取得"""
    memory = psutil.virtual_memory()
    gpu_info = {}
    
    if torch.cuda.is_available():
        gpu_info = {
            'allocated': torch.cuda.memory_allocated() / 1024**2,
            'reserved': torch.cuda.memory_reserved() / 1024**2,
            'total': torch.cuda.get_device_properties(0).total_memory / 1024**2
        }
    
    return {
        'cpu_percent': memory.percent,
        'cpu_used_gb': memory.used / 1024**3,
        'cpu_total_gb': memory.total / 1024**3,
        'gpu': gpu_info
    }

def run_facefusion_subprocess():
    """サブプロセスでFaceFusionを実行"""
    print("🚀 サブプロセスでFaceFusion実行開始")
    
    # 環境変数設定
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = "../facefusion_env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:../facefusion_env/lib/python3.10/site-packages/nvidia/cudnn/lib:../facefusion_env/lib/python3.10/site-packages/nvidia/cublas/lib:../facefusion_env/lib/python3.10/site-packages/nvidia/cufft/lib:" + env.get('LD_LIBRARY_PATH', '')
    
    # FaceFusionコマンド
    cmd = [
        'bash', '-c',
        'source ../facefusion_env/bin/activate && python facefusion.py headless-run '
        '--source-paths ../input/source_face.jpg '
        '--target-path ../input/target_video_3s.mp4 '
        '--output-path ../output/facefusion/test_subprocess.mp4 '
        '--execution-providers cuda '
        '--face-swapper-model inswapper_128 '
        '--processors face_swapper'
    ]
    
    # 開始前メモリ状況
    mem_before = get_memory_info()
    print(f"開始前メモリ: CPU={mem_before['cpu_percent']:.1f}% ({mem_before['cpu_used_gb']:.1f}GB)")
    if mem_before['gpu']:
        print(f"開始前GPU: {mem_before['gpu']['allocated']:.1f}MB allocated, {mem_before['gpu']['reserved']:.1f}MB reserved")
    
    start_time = time.time()
    
    try:
        # サブプロセス実行（タイムアウト設定）
        print("サブプロセス開始...")
        process = subprocess.Popen(
            cmd,
            cwd='/home/adama/wav2lip-project/facefusion',
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # プロセスグループ作成
        )
        
        # プロセス監視
        monitor_process(process)
        
        # プロセス完了待機
        stdout, stderr = process.communicate(timeout=300)  # 5分タイムアウト
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ サブプロセス完了: {processing_time:.1f}秒")
        print(f"終了コード: {process.returncode}")
        
        # 標準出力の最後の部分を表示
        if stdout:
            lines = stdout.strip().split('\n')
            print("実行結果（最後の10行）:")
            for line in lines[-10:]:
                print(f"  {line}")
        
        if stderr and process.returncode != 0:
            print("エラー出力:")
            print(stderr[-1000:])  # 最後の1000文字
            
    except subprocess.TimeoutExpired:
        print("⏰ タイムアウト - プロセス強制終了")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except:
            process.kill()
        return False
        
    except Exception as e:
        print(f"❌ サブプロセス実行エラー: {e}")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except:
            pass
        return False
    
    finally:
        # 完了後メモリ状況
        time.sleep(2)  # プロセス終了後の安定化待機
        mem_after = get_memory_info()
        print(f"完了後メモリ: CPU={mem_after['cpu_percent']:.1f}% ({mem_after['cpu_used_gb']:.1f}GB)")
        if mem_after['gpu']:
            print(f"完了後GPU: {mem_after['gpu']['allocated']:.1f}MB allocated, {mem_after['gpu']['reserved']:.1f}MB reserved")
        
        # メモリ使用量変化
        cpu_diff = mem_after['cpu_percent'] - mem_before['cpu_percent']
        print(f"CPUメモリ変化: {cpu_diff:+.1f}%")
        
        if mem_before['gpu'] and mem_after['gpu']:
            gpu_diff = mem_after['gpu']['allocated'] - mem_before['gpu']['allocated']
            print(f"GPUメモリ変化: {gpu_diff:+.1f}MB")
    
    return process.returncode == 0

def monitor_process(process):
    """プロセス実行中のメモリ監視"""
    print("📊 プロセス監視開始...")
    
    try:
        psutil_process = psutil.Process(process.pid)
        max_memory = 0
        
        while process.poll() is None:  # プロセスが実行中
            try:
                # プロセスメモリ使用量
                mem_info = psutil_process.memory_info()
                current_memory = mem_info.rss / 1024**2  # MB
                max_memory = max(max_memory, current_memory)
                
                # システム全体のメモリ
                system_mem = psutil.virtual_memory().percent
                
                print(f"監視: プロセスメモリ={current_memory:.1f}MB (最大{max_memory:.1f}MB), システム={system_mem:.1f}%", end='\r')
                
                # OOM警告
                if system_mem > 90:
                    print(f"\n⚠️  警告: システムメモリ使用量が{system_mem:.1f}%に達しています")
                
                time.sleep(2)
                
            except psutil.NoSuchProcess:
                break
                
    except Exception as e:
        print(f"\n監視エラー: {e}")
    
    print(f"\n📊 最大メモリ使用量: {max_memory:.1f}MB")

def test_multiple_runs():
    """複数回実行テスト（メモリリーク検出）"""
    print("\n🔄 複数回実行テスト開始")
    
    results = []
    for i in range(3):
        print(f"\n--- 実行 {i+1}/3 ---")
        
        mem_before = get_memory_info()
        success = run_facefusion_subprocess()
        mem_after = get_memory_info()
        
        results.append({
            'run': i+1,
            'success': success,
            'cpu_before': mem_before['cpu_percent'],
            'cpu_after': mem_after['cpu_percent'],
            'gpu_before': mem_before['gpu']['allocated'] if mem_before['gpu'] else 0,
            'gpu_after': mem_after['gpu']['allocated'] if mem_after['gpu'] else 0
        })
        
        time.sleep(5)  # 実行間隔
    
    # 結果分析
    print("\n📈 複数回実行結果:")
    for result in results:
        cpu_diff = result['cpu_after'] - result['cpu_before']
        gpu_diff = result['gpu_after'] - result['gpu_before']
        status = "✅" if result['success'] else "❌"
        print(f"実行{result['run']}: {status} CPU変化={cpu_diff:+.1f}%, GPU変化={gpu_diff:+.1f}MB")
    
    # メモリリーク検出
    cpu_trend = [r['cpu_after'] for r in results]
    gpu_trend = [r['gpu_after'] for r in results]
    
    if len(cpu_trend) >= 2:
        cpu_increase = cpu_trend[-1] - cpu_trend[0]
        if cpu_increase > 5:  # 5%以上の増加
            print("⚠️  CPUメモリリークの可能性あり")
        else:
            print("✅ CPUメモリリークなし")
    
    if len(gpu_trend) >= 2 and any(gpu_trend):
        gpu_increase = gpu_trend[-1] - gpu_trend[0]
        if gpu_increase > 100:  # 100MB以上の増加
            print("⚠️  GPUメモリリークの可能性あり")
        else:
            print("✅ GPUメモリリークなし")

def main():
    print("🧪 FaceFusion サブプロセステスト")
    print("=" * 60)
    
    # 単回実行テスト
    print("1️⃣ 単回実行テスト")
    success = run_facefusion_subprocess()
    
    if not success:
        print("❌ 単回実行失敗 - 複数回テストをスキップ")
        return
    
    # 複数回実行テスト
    print("\n2️⃣ 複数回実行テスト（メモリリーク検出）")
    test_multiple_runs()
    
    print("\n🎯 サブプロセステスト完了！")

if __name__ == "__main__":
    main()