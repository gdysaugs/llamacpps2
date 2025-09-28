#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from wav2lip_model import Wav2LipModel
from processor import GFPGANEnhancer

def main():
    parser = argparse.ArgumentParser(description='Wav2Lip ONNX - Generate lip-sync videos')
    parser.add_argument('-v', '--video', required=True, help='Path to input video file')
    parser.add_argument('-a', '--audio', required=True, help='Path to input audio file')
    parser.add_argument('-o', '--output', required=True, help='Path to output video file')
    parser.add_argument('--model-path', default='models', help='Path to models directory')
    parser.add_argument('--use-gfpgan', action='store_true', help='Use GFPGAN for face enhancement')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
        
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)
        
    # Get absolute model path
    script_dir = Path(__file__).parent
    if os.path.isabs(args.model_path):
        model_path = Path(args.model_path)
    else:
        # Relative to project root, not script directory
        model_path = script_dir.parent / args.model_path
    model_path = model_path.resolve()
    
    if not model_path.exists():
        print(f"Error: Model directory not found: {model_path}")
        sys.exit(1)
        
    print("=" * 50)
    print("Wav2Lip ONNX HQ")
    print("=" * 50)
    
    # Check GPU availability
    if args.device == 'cuda':
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' not in providers:
                print("Warning: CUDA not available, falling back to CPU")
                args.device = 'cpu'
            else:
                print(f"✓ Using GPU (CUDA)")
        except:
            args.device = 'cpu'
    
    if args.device == 'cpu':
        print("Using CPU (this will be slower)")
    
    # Initialize Wav2Lip model
    model = Wav2LipModel(model_path, device=args.device)
    
    # Initialize GFPGAN if requested
    enhancer = None
    if args.use_gfpgan:
        print("\nInitializing GFPGAN enhancer...")
        enhancer = GFPGANEnhancer(model_path, device=args.device)
    
    # Process video
    try:
        model.process_video(
            video_path=args.video,
            audio_path=args.audio,
            output_path=args.output
        )
        print("\n✓ Processing complete!")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()