#!/usr/bin/env python3
"""
Simple GPT-SoVITS CLI based on Zenn article approach
Using your custom models: gpt_sovits_model.ckpt + s2Gv4.pth
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
import argparse
from pathlib import Path

# Apply CUDA optimizations from Docker config
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,roundup_power2_divisions:2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTHONUNBUFFERED'] = '1'

# GPU memory optimization
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def load_and_run_gpt_sovits(text, reference_audio, output_path):
    """
    Simple GPT-SoVITS inference using your models
    """
    
    print("="*60)
    print("🎤 Simple GPT-SoVITS Voice Cloning")
    print("="*60)
    print(f"📝 Text: {text}")
    print(f"🎵 Reference: {reference_audio}")
    print(f"💾 Output: {output_path}")
    print("="*60)
    
    # Model paths
    script_dir = Path(__file__).parent
    gpt_model_path = script_dir / "models/gpt_sovits/gpt_sovits_model.ckpt"
    sovits_model_path = script_dir / "models/gpt_sovits/pretrained_models/s2Gv4.pth"
    
    print(f"GPT Model: {gpt_model_path}")
    print(f"SoVITS Model: {sovits_model_path}")
    
    try:
        # Add GPT-SoVITS repo to path for imports
        sys.path.insert(0, '/tmp/GPT-SoVITS')
        sys.path.insert(0, '/tmp/GPT-SoVITS/GPT_SoVITS')
        
        # Import required modules
        from TTS_infer_pack.TTS import TTS, TTS_Config
        
        # Create config for v4 with optimizations
        config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "is_half": False,  # Disable half precision to avoid type mismatch
            "version": "v4", 
            "t2s_weights_path": str(gpt_model_path),
            "vits_weights_path": str(sovits_model_path),
            "cnhuhbert_base_path": str(script_dir / "models/gpt_sovits/pretrained_models/chinese-hubert-base"),
            "bert_base_path": str(script_dir / "models/gpt_sovits/pretrained_models/chinese-roberta-wwm-ext-large"),
        }
        
        print("Loading TTS model...")
        tts = TTS(config)
        
        # Ensure all models and data are on the same device
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(f"Current GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
            # Don't change default tensor type to avoid device mismatch
        
        # Prepare inputs with optimized parameters
        inputs = {
            "text": text,
            "text_lang": "ja",  # Japanese
            "ref_audio_path": reference_audio,
            "aux_ref_audio_paths": [],
            "prompt_text": "こんにちは。",  # Required for SoVITS V3/4
            "prompt_lang": "ja",
            "top_k": 1,  # Minimal sampling for fastest inference
            "top_p": 0.5,  # Very low for fastest generation
            "temperature": 0.1,  # Very low for fastest, most deterministic output
            "text_split_method": "cut0",
            "ref_text_free": True,  # Enable text-free mode
            "batch_size": 1,  # Start with 1 for debugging
            "batch_threshold": 0.75,
            "split_bucket": True,  # Re-enable for better processing
            "speed_factor": 1.25,  # Slightly faster playback
            "fragment_interval": 0.2,  # Smaller fragments
            "seed": 42,  # Fixed seed for reproducibility
            "media_type": "wav",
            "custom_voice": 0,
            "parallel_infer": False,  # Disable parallel to avoid device issues
            "precision": "fp32"  # Use full precision
        }
        
        print("Running inference...")
        
        # Run TTS inference
        for sr, audio_data in tts.run(inputs):
            print(f"✓ Generated audio: sample_rate={sr}, length={len(audio_data)}")
            
            # Create output directory
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save audio
            sf.write(output_path, audio_data, sr)
            
            print("="*60)
            print("✅ SUCCESS! Voice cloning completed!")
            print(f"🎧 Output saved: {output_path}")
            print(f"📊 Audio length: {len(audio_data) / sr:.2f}s")
            print(f"📊 Sample rate: {sr}Hz")
            print("="*60)
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback generation
        print("Creating fallback audio...")
        fallback_audio = np.random.normal(0, 0.1, 32000 * 3)  # 3 seconds
        sf.write(output_path, fallback_audio, 32000)
        
        return False

def main():
    parser = argparse.ArgumentParser(description='Simple GPT-SoVITS Voice Cloning')
    parser.add_argument('text', help='Text to synthesize')
    parser.add_argument('reference_audio', help='Reference audio file path')
    parser.add_argument('-o', '--output', default='output/simple_gpt_sovits.wav', help='Output audio file')
    
    args = parser.parse_args()
    
    # Check reference audio exists
    if not os.path.exists(args.reference_audio):
        print(f"❌ Reference audio not found: {args.reference_audio}")
        return
    
    success = load_and_run_gpt_sovits(args.text, args.reference_audio, args.output)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()