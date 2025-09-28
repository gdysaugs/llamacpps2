#!/usr/bin/env python3
"""
GPT-SoVITS Voice Cloning CLI Interface
Voice cloning with reference audio - text-free mode supported
"""

import sys
import os
import argparse
from pathlib import Path
import shutil

# Add src directory to path (EXE compatible)
if getattr(sys, 'frozen', False):
    # EXE実行時（PyInstaller/Nuitka）
    script_dir = Path(sys.executable).parent
else:
    # 通常のPython実行時
    script_dir = Path(__file__).parent

src_dir = script_dir / 'src'
sys.path.insert(0, str(src_dir))

from gpt_sovits_engine import GPTSoVITSEngine, SimplifiedGPTSoVITS
from real_gpt_sovits import RealGPTSoVITS
from real_gpt_sovits_v2 import RealGPTSoVITSV2


def main():
    parser = argparse.ArgumentParser(
        description='GPT-SoVITS Voice Cloning - Generate speech with cloned voice',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (text-free mode)
  python gpt_sovits_inference.py "Hello world" reference.wav
  
  # Specify output file
  python gpt_sovits_inference.py "こんにちは" voice.wav -o output/cloned.wav
  
  # Adjust speech speed
  python gpt_sovits_inference.py "Text to speak" ref.wav --speed 1.2
  
  # Use specific language
  python gpt_sovits_inference.py "你好" ref.wav --language zh
  
  # With reference text (better quality)
  python gpt_sovits_inference.py "New text" ref.wav --ref-text "Original text"
        """
    )
    
    # Required arguments
    parser.add_argument('text', 
                        help='Text to synthesize with cloned voice')
    parser.add_argument('reference_audio', 
                        help='Path to reference audio file for voice cloning')
    
    # Optional arguments
    parser.add_argument('-o', '--output', 
                        default='output/gpt_sovits_result.wav',
                        help='Output audio file path (default: output/gpt_sovits_result.wav)')
    parser.add_argument('--model', 
                        help='Path to GPT-SoVITS model checkpoint (.ckpt file)')
    parser.add_argument('--device', 
                        default='cuda', 
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--language', 
                        default='ja', 
                        choices=['ja', 'en', 'zh', 'ko'],
                        help='Target language (default: ja)')
    parser.add_argument('--speed', 
                        type=float, 
                        default=1.0,
                        help='Speech speed factor (default: 1.0)')
    parser.add_argument('--temperature', 
                        type=float, 
                        default=0.3,
                        help='Sampling temperature for voice variation (default: 0.3)')
    parser.add_argument('--ref-text', 
                        help='Reference text for the audio (optional, for better quality)')
    parser.add_argument('--text-free', 
                        action='store_true', 
                        default=True,
                        help='Use text-free mode (default: True)')
    parser.add_argument('--simplified', 
                        action='store_true',
                        help='Use simplified implementation (faster but lower quality)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*60)
    print("🎤 GPT-SoVITS Voice Cloning")
    print("="*60)
    print(f"📝 Text: {args.text}")
    print(f"🎵 Reference: {args.reference_audio}")
    print(f"💾 Output: {args.output}")
    print(f"⚙️  Device: {args.device}")
    print(f"🌐 Language: {args.language}")
    print(f"⚡ Speed: {args.speed}x")
    print(f"🔥 Temperature: {args.temperature}")
    print(f"📄 Text-free mode: {args.ref_text is None}")
    if args.simplified:
        print(f"⚡ Mode: Simplified (fast)")
    print("="*60)
    
    try:
        # Check if reference audio exists
        if not os.path.exists(args.reference_audio):
            # Try Windows path conversion for WSL
            win_path = args.reference_audio.replace('C:', '/mnt/c').replace('\\', '/')
            if os.path.exists(win_path):
                args.reference_audio = win_path
            else:
                raise FileNotFoundError(f"Reference audio not found: {args.reference_audio}")
        
        # Determine model path
        if args.model:
            model_path = args.model
            # Handle Windows path in WSL
            if not os.path.exists(model_path):
                model_path = model_path.replace('C:', '/mnt/c').replace('\\', '/')
        else:
            # Use default model location
            models_dir = script_dir / 'models' / 'gpt_sovits'
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Look for model file
            model_files = list(models_dir.glob('*.ckpt')) + list(models_dir.glob('*.pth'))
            
            if model_files:
                model_path = str(model_files[0])
                print(f"✓ Using model: {model_path}")
            else:
                # Try to copy from Downloads if specified
                download_path = "/mnt/c/Users/adama/Downloads/gpt_sovits_models_hscene-e17.ckpt"
                if os.path.exists(download_path):
                    target_path = models_dir / "gpt_sovits_model.ckpt"
                    print(f"📥 Copying model from Downloads...")
                    shutil.copy2(download_path, target_path)
                    model_path = str(target_path)
                    print(f"✓ Model copied to: {model_path}")
                else:
                    model_path = None
        
        # Create output directory
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.simplified:
            # Use simplified implementation
            print("\n🚀 Using simplified implementation...")
            engine = SimplifiedGPTSoVITS(
                model_dir=str(script_dir / 'models' / 'gpt_sovits'),
                device=args.device
            )
            
            # Run voice cloning
            engine.clone_voice(
                text=args.text,
                reference_audio=args.reference_audio,
                output_path=args.output,
                speed=args.speed,
                temperature=args.temperature
            )
        else:
            # Use real GPT-SoVITS V2 implementation with official code base
            print(f"\n🧠 Using real GPT-SoVITS V2 with official implementation...")
            engine = RealGPTSoVITSV2(
                model_dir=str(script_dir / 'models' / 'gpt_sovits'),
                device=args.device
            )
            
            # Run voice cloning
            engine.clone_voice(
                text=args.text,
                reference_audio=args.reference_audio,
                output_path=args.output,
                reference_text=args.ref_text,
                speed=args.speed,
                temperature=args.temperature
            )
        
        print("="*60)
        print("✅ SUCCESS! Voice cloning completed!")
        print(f"🎧 Output saved: {args.output}")
        print("="*60)
        
        # Play audio if possible (optional)
        if os.path.exists(args.output):
            file_size = os.path.getsize(args.output) / 1024  # KB
            print(f"📊 File size: {file_size:.1f} KB")
        
    except FileNotFoundError as e:
        print("="*60)
        print(f"❌ ERROR: {e}")
        print("="*60)
        print("\nPlease ensure:")
        print("1. Reference audio file exists")
        print("2. Model file is in models/gpt_sovits/ directory")
        print("3. For WSL, use /mnt/c/ paths for Windows files")
        sys.exit(1)
        
    except Exception as e:
        import traceback
        print("="*60)
        print("❌ ERROR occurred during processing:")
        print(f"Error: {e}")
        print("\nTraceback:")
        print(traceback.format_exc())
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()