#!/usr/bin/env python3
"""
Download GPT-SoVITS models from HuggingFace
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil

def download_gpt_sovits_models():
    """Download essential GPT-SoVITS models"""
    
    # Model directory
    models_dir = Path("models/gpt_sovits/pretrained_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Repository info
    repo_id = "lj1995/GPT-SoVITS"
    
    # Essential model files for Japanese TTS
    model_files = [
        {
            "filename": "s1v3.ckpt",
            "local_path": models_dir / "s1v3.ckpt",
            "description": "GPT semantic model (155 MB)",
            "essential": True
        },
        {
            "filename": "s2Gv3.pth", 
            "local_path": models_dir / "s2Gv3.pth",
            "description": "SoVITS generator model (769 MB)",
            "essential": True
        },
        {
            "filename": "s2D488k.pth",
            "local_path": models_dir / "s2D488k.pth", 
            "description": "SoVITS discriminator model (93.5 MB)",
            "essential": False
        }
    ]
    
    # Additional models
    additional_models = [
        {
            "filename": "chinese-hubert-base/pytorch_model.bin",
            "local_path": models_dir / "chinese-hubert-base" / "pytorch_model.bin",
            "description": "SSL feature extractor",
            "essential": True
        },
        {
            "filename": "chinese-roberta-wwm-ext-large/pytorch_model.bin", 
            "local_path": models_dir / "chinese-roberta-wwm-ext-large" / "pytorch_model.bin",
            "description": "BERT model for text processing",
            "essential": True
        }
    ]
    
    print("="*60)
    print("📥 Downloading GPT-SoVITS Models")
    print("="*60)
    
    # Download essential models first
    for model in model_files:
        if model["essential"] or not model["local_path"].exists():
            print(f"\n📦 Downloading {model['description']}...")
            print(f"   File: {model['filename']}")
            
            try:
                # Create parent directory
                model["local_path"].parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                downloaded_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=model["filename"],
                    local_dir=None,  # Use cache
                    cache_dir=None
                )
                
                # Move to our models directory
                shutil.copy2(downloaded_file, model["local_path"])
                
                print(f"   ✅ Downloaded: {model['local_path']}")
                print(f"   📊 Size: {model['local_path'].stat().st_size / (1024*1024):.1f} MB")
                
            except Exception as e:
                print(f"   ❌ Failed to download {model['filename']}: {e}")
                if model["essential"]:
                    print("   🚨 This is an essential file - inference may not work")
        else:
            print(f"✓ Already exists: {model['filename']}")
    
    # Download additional models (non-essential)
    print(f"\n📦 Downloading additional models...")
    for model in additional_models:
        if not model["local_path"].exists():
            try:
                print(f"   Downloading {model['description']}...")
                
                # Create directory structure
                model["local_path"].parent.mkdir(parents=True, exist_ok=True)
                
                downloaded_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=model["filename"],
                    local_dir=None
                )
                
                shutil.copy2(downloaded_file, model["local_path"])
                print(f"   ✅ Downloaded: {model['filename']}")
                
            except Exception as e:
                print(f"   ⚠️  Optional download failed: {model['filename']}: {e}")
        else:
            print(f"   ✓ Already exists: {model['filename']}")
    
    print("\n" + "="*60)
    print("📋 Model Download Summary")
    print("="*60)
    
    # Check what we have
    essential_files = []
    optional_files = []
    
    for model in model_files:
        if model["local_path"].exists():
            size_mb = model["local_path"].stat().st_size / (1024*1024)
            status = "✅ Ready"
            (essential_files if model["essential"] else optional_files).append(
                f"  {model['filename']}: {size_mb:.1f}MB - {status}"
            )
        else:
            status = "❌ Missing"
            (essential_files if model["essential"] else optional_files).append(
                f"  {model['filename']}: {status}"
            )
    
    print("Essential Models:")
    for file_info in essential_files:
        print(file_info)
    
    if optional_files:
        print("\nOptional Models:")
        for file_info in optional_files:
            print(file_info)
    
    # Calculate total size
    total_size = 0
    for model in model_files:
        if model["local_path"].exists():
            total_size += model["local_path"].stat().st_size
    
    print(f"\n📊 Total downloaded: {total_size / (1024*1024*1024):.2f} GB")
    
    # Check if we have minimum requirements
    essential_missing = [m for m in model_files if m["essential"] and not m["local_path"].exists()]
    
    if essential_missing:
        print("\n🚨 WARNING: Missing essential models!")
        for model in essential_missing:
            print(f"   - {model['filename']}")
        print("   GPT-SoVITS inference may not work properly.")
        return False
    else:
        print("\n🎉 All essential models downloaded successfully!")
        print("   Ready for GPT-SoVITS inference.")
        return True

if __name__ == "__main__":
    success = download_gpt_sovits_models()
    exit(0 if success else 1)