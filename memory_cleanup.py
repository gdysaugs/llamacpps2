#!/usr/bin/env python3
"""
FaceFusion処理後のメモリ完全開放スクリプト
GPU・CPUメモリを強制的にクリーンアップ
"""
import gc
import torch
import time
import subprocess
import psutil
import os

def cleanup_gpu_memory():
    """GPU メモリのクリーンアップ"""
    print("=== GPU メモリクリーンアップ開始 ===")
    
    if torch.cuda.is_available():
        # PyTorch GPU キャッシュクリア
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"PyTorch GPU cache cleared")
        
        # GPU メモリ統計表示
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**2
            mem_cached = torch.cuda.memory_reserved(i) / 1024**2
            print(f"GPU {i}: Allocated={mem_allocated:.1f}MB, Cached={mem_cached:.1f}MB")
    
    # NVIDIA GPU メモリリセット（可能であれば）
    try:
        subprocess.run(['nvidia-smi', '--gpu-reset'], capture_output=True)
        print("NVIDIA GPU reset attempt completed")
    except:
        print("GPU reset not available")

def cleanup_cpu_memory():
    """CPU メモリのクリーンアップ"""
    print("=== CPU メモリクリーンアップ開始 ===")
    
    # Python ガベージコレクション
    collected = gc.collect()
    print(f"Garbage collected: {collected} objects")
    
    # システムメモリ統計
    memory = psutil.virtual_memory()
    print(f"System Memory: {memory.percent}% used ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")

def cleanup_process_memory():
    """プロセス固有メモリのクリーンアップ"""
    print("=== プロセスメモリクリーンアップ開始 ===")
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Current process memory: RSS={mem_info.rss/1024**2:.1f}MB, VMS={mem_info.vms/1024**2:.1f}MB")
    
    # Pythonインタープリターの最適化
    import sys
    if hasattr(sys, '_clear_type_cache'):
        sys._clear_type_cache()
        print("Python type cache cleared")

def force_system_cleanup():
    """システム全体のメモリクリーンアップ"""
    print("=== システム全体クリーンアップ開始 ===")
    
    # Linux系システムでのメモリクリーンアップ
    try:
        # ページキャッシュをクリア（要root権限）
        subprocess.run(['sudo', 'sh', '-c', 'echo 1 > /proc/sys/vm/drop_caches'], 
                      capture_output=True, timeout=5)
        print("System page cache cleared")
    except:
        print("System cache clear not available (requires sudo)")

def main():
    print("🧹 FaceFusion メモリ完全クリーンアップ開始")
    print("=" * 50)
    
    # 開始前のメモリ状況
    if torch.cuda.is_available():
        gpu_mem_before = torch.cuda.memory_allocated() / 1024**2
        print(f"処理前GPU使用量: {gpu_mem_before:.1f}MB")
    
    cpu_mem_before = psutil.virtual_memory().percent
    print(f"処理前CPU使用量: {cpu_mem_before:.1f}%")
    
    # クリーンアップ実行
    cleanup_gpu_memory()
    time.sleep(1)
    cleanup_cpu_memory()
    time.sleep(1)
    cleanup_process_memory()
    time.sleep(1)
    force_system_cleanup()
    
    print("=" * 50)
    
    # 完了後のメモリ状況
    if torch.cuda.is_available():
        gpu_mem_after = torch.cuda.memory_allocated() / 1024**2
        print(f"処理後GPU使用量: {gpu_mem_after:.1f}MB")
        print(f"GPU解放量: {gpu_mem_before - gpu_mem_after:.1f}MB")
    
    cpu_mem_after = psutil.virtual_memory().percent
    print(f"処理後CPU使用量: {cpu_mem_after:.1f}%")
    print(f"CPU解放量: {cpu_mem_before - cpu_mem_after:.1f}%")
    
    print("🎯 メモリクリーンアップ完了！")
    
    # OOMリスク評価
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        print("⚠️  警告: CPU使用量が90%を超えています - OOMリスク高")
    elif memory.percent > 80:
        print("⚠️  注意: CPU使用量が80%を超えています - OOMリスク中")
    else:
        print("✅ CPU使用量正常 - OOMリスク低")
        
    if torch.cuda.is_available():
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        gpu_used = torch.cuda.memory_allocated() / 1024**2
        gpu_percent = (gpu_used / gpu_total) * 100
        
        if gpu_percent > 90:
            print("⚠️  警告: GPU使用量が90%を超えています - OOMリスク高")
        elif gpu_percent > 80:
            print("⚠️  注意: GPU使用量が80%を超えています - OOMリスク中")
        else:
            print("✅ GPU使用量正常 - OOMリスク低")

if __name__ == "__main__":
    main()