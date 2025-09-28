#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from wav2lip_inference import Wav2LipONNXInference

def main():
    parser = argparse.ArgumentParser(description='Wav2Lip ONNX HQ - High Quality Lip-sync')
    parser.add_argument('-v', '--video', required=True, help='Path to input video file')
    parser.add_argument('-a', '--audio', required=True, help='Path to input audio file')
    parser.add_argument('-o', '--output', required=True, help='Path to output video file')
    parser.add_argument('--model-path', default='models', help='Path to models directory')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
        
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)
    
    # Get model path
    script_dir = Path(__file__).parent
    if os.path.isabs(args.model_path):
        model_path = Path(args.model_path)
    else:
        model_path = script_dir.parent / args.model_path
    model_path = model_path.resolve()
    
    if not model_path.exists():
        print(f"Error: Model directory not found: {model_path}")
        sys.exit(1)
    
    print("=" * 50)
    print("Wav2Lip ONNX HQ")
    print("=" * 50)
    
    # Initialize inference
    try:
        inference = Wav2LipONNXInference(str(model_path), device=args.device)
        
        # Run inference
        inference.inference(
            video_path=args.video,
            audio_path=args.audio,
            output_path=args.output
        )
        
        print("\n✓ Processing complete!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()