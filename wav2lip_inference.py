#!/usr/bin/env python3
import sys
import os
import argparse
from pathlib import Path

# Add src directory to path (EXE compatible)
if getattr(sys, 'frozen', False):
    # EXE実行時（PyInstaller/Nuitka）
    script_dir = Path(sys.executable).parent
else:
    # 通常のPython実行時
    script_dir = Path(__file__).parent

src_dir = script_dir / 'src'
sys.path.insert(0, str(src_dir))

from wav2lip_final import Wav2LipFinal

def main():
    parser = argparse.ArgumentParser(description='Wav2Lip ONNX High Quality Inference')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('audio_path', help='Path to input audio file')
    parser.add_argument('-o', '--output', default='output/result.mp4', 
                        help='Output video path (default: output/result.mp4)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--gfpgan', action='store_true', default=True,
                        help='Use GFPGAN enhancement (default: enabled)')
    parser.add_argument('--no-gfpgan', action='store_true',
                        help='Disable GFPGAN enhancement')
    parser.add_argument('--gfpgan-blend', type=float, default=0.35,
                        help='GFPGAN blend ratio 0.0-1.0 (default: 0.35)')
    parser.add_argument('--gfpgan-mouth-only', action='store_true',
                        help='Apply GFPGAN only to mouth region instead of full face')
    parser.add_argument('--resize-factor', type=int, default=1,
                        help='Resize factor for processing (default: 1)')
    parser.add_argument('--use-384-model', action='store_true',
                        help='Use 384x384 model instead of 96x96')
    
    args = parser.parse_args()
    
    # Handle GFPGAN settings
    use_gfpgan = args.gfpgan and not args.no_gfpgan
    
    print("="*60)
    print("🎬 Wav2Lip ONNX High Quality Inference")
    print("="*60)
    print(f"📹 Video: {args.video_path}")
    print(f"🎵 Audio: {args.audio_path}")
    print(f"💾 Output: {args.output}")
    print(f"⚡ Device: {args.device}")
    print(f"✨ GFPGAN: {'Enabled' if use_gfpgan else 'Disabled'}")
    if use_gfpgan:
        print(f"🎚️  GFPGAN Blend: {args.gfpgan_blend:.1%}")
        print(f"👄 GFPGAN Mode: {'Mouth Only' if args.gfpgan_mouth_only else 'Full Face'}")
    print(f"📐 Resize Factor: {args.resize_factor}")
    print(f"🧠 Model: {'384x384' if args.use_384_model else '96x96 GAN'}")
    print("="*60)
    
    try:
        # Initialize Wav2Lip with specified parameters (EXE compatible paths)
        if getattr(sys, 'frozen', False):
            # EXE実行時はexeと同階層のmodelsディレクトリ
            models_dir = script_dir / 'models'
        else:
            # 通常のPython実行時
            models_dir = script_dir / 'models'
        wav2lip = Wav2LipFinal(
            str(models_dir),
            device=args.device,
            use_gfpgan=use_gfpgan,
            gfpgan_blend=args.gfpgan_blend,
            use_gan_model=True,  # Always use GAN model for quality
            use_384_model=args.use_384_model,
            resize_factor=args.resize_factor,
            gfpgan_mouth_only=args.gfpgan_mouth_only
        )
        
        # Run inference
        wav2lip.inference(
            video_path=args.video_path,
            audio_path=args.audio_path,
            output_path=args.output
        )
        
        print("="*60)
        print("✅ SUCCESS! Lip-sync generation completed!")
        print(f"🎥 Output saved: {args.output}")
        print("="*60)
        
    except Exception as e:
        import traceback
        print("="*60)
        print("❌ ERROR occurred during processing:")
        print(f"Error: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()