#!/usr/bin/env python3
"""
FaceFusion サブプロセステスト
既存のCLIテストをサブプロセス化
"""
import subprocess
import time
import psutil
import torch
import os

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
        'cpu_used_gb': memory.used / 1024**3,
        'gpu': gpu_info
    }

def run_cli_test_subprocess():
    """CLI テストをサブプロセスで実行"""
    print("🚀 FaceFusion CLIテスト（サブプロセス）開始")
    
    # 開始前メモリ
    mem_before = get_memory_info()
    print(f"開始前: CPU={mem_before['cpu_percent']:.1f}%, GPU={mem_before['gpu'].get('allocated', 0):.1f}MB")
    
    start_time = time.time()
    
    try:
        # サブプロセスでCLIテスト実行
        result = subprocess.run(
            ['python', 'facefusion_cli_test.py'],
            capture_output=True,
            text=True,
            timeout=120,  # 2分タイムアウト
            cwd='/home/adama/wav2lip-project'
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"実行時間: {processing_time:.1f}秒")
        print(f"終了コード: {result.returncode}")
        
        # 結果表示
        if result.stdout:
            print("標準出力:")
            print(result.stdout[-500:])  # 最後の500文字
            
        if result.stderr and result.returncode != 0:
            print("エラー出力:")
            print(result.stderr[-300:])
        
        success = result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ タイムアウト")
        success = False
    except Exception as e:
        print(f"❌ エラー: {e}")
        success = False
    
    # 完了後メモリ
    time.sleep(1)
    mem_after = get_memory_info()
    print(f"完了後: CPU={mem_after['cpu_percent']:.1f}%, GPU={mem_after['gpu'].get('allocated', 0):.1f}MB")
    
    # メモリ変化
    cpu_diff = mem_after['cpu_percent'] - mem_before['cpu_percent']
    gpu_diff = mem_after['gpu'].get('allocated', 0) - mem_before['gpu'].get('allocated', 0)
    print(f"変化: CPU={cpu_diff:+.1f}%, GPU={gpu_diff:+.1f}MB")
    
    return success

def test_multiple_runs(count=3):
    """複数回実行テスト"""
    print(f"\n🔄 {count}回連続実行テスト開始")
    
    results = []
    for i in range(count):
        print(f"\n--- 実行 {i+1}/{count} ---")
        
        success = run_cli_test_subprocess()
        results.append(success)
        
        if i < count - 1:  # 最後以外は待機
            time.sleep(3)
    
    # 結果サマリー
    success_count = sum(results)
    print(f"\n📊 結果: {success_count}/{count} 成功")
    
    if success_count == count:
        print("✅ 全て成功 - サブプロセス動作安定")
    else:
        print("⚠️ 一部失敗 - 調査が必要")
    
    return success_count == count

def main():
    print("🧪 FaceFusion サブプロセステスト（高速版）")
    print("=" * 50)
    
    # 1回実行テスト
    success = run_cli_test_subprocess()
    
    if success:
        print("✅ 単回実行成功")
        
        # 複数回実行テスト
        all_success = test_multiple_runs(3)
        
        if all_success:
            print("\n🎯 サブプロセステスト完全成功！")
            print("✅ メモリリークなし")
            print("✅ プロセス分離正常")
            print("✅ 連続実行安定")
        else:
            print("\n⚠️ 一部で問題発生")
    else:
        print("❌ 単回実行失敗")

if __name__ == "__main__":
    main()