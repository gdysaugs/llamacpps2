#!/usr/bin/env python3
"""
Advanced GPT-SoVITS Voice Cloning Test CLI
Based on the official GPT-SoVITS api_v2.py implementation
"""

import argparse
import requests
import json
import base64
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

class GPTSoVITSClient:
    """Client for GPT-SoVITS API v2"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 9885):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        
    def check_health(self) -> bool:
        """Check if API server is running"""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def set_gpt_weights(self, weights_path: str) -> bool:
        """Set GPT model weights"""
        try:
            response = self.session.post(
                f"{self.base_url}/set_gpt_weights",
                json={"weights_path": weights_path}
            )
            return response.status_code == 200
        except:
            return False
    
    def set_sovits_weights(self, weights_path: str) -> bool:
        """Set SoVITS model weights"""
        try:
            response = self.session.post(
                f"{self.base_url}/set_sovits_weights", 
                json={"weights_path": weights_path}
            )
            return response.status_code == 200
        except:
            return False
    
    def voice_clone(
        self,
        text: str,
        text_lang: str = "ja",
        ref_audio_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        prompt_lang: str = "ja",
        top_k: int = 5,
        top_p: float = 1.0,
        temperature: float = 1.0,
        text_split_method: str = "cut5",
        batch_size: int = 1,
        batch_threshold: float = 0.75,
        split_bucket: bool = True,
        speed_factor: float = 1.0,
        fragment_interval: float = 0.3,
        seed: int = -1,
        media_type: str = "wav",
        streaming_mode: bool = False,
        parallel_infer: bool = True,
        repetition_penalty: float = 1.35
    ) -> Optional[bytes]:
        """
        Perform voice cloning with advanced parameters
        """
        
        # Build request payload
        payload = {
            "text": text,
            "text_lang": text_lang,
            "prompt_lang": prompt_lang,
            "text_split_method": text_split_method,
            "batch_size": batch_size,
            "batch_threshold": batch_threshold,
            "split_bucket": split_bucket,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "speed_factor": speed_factor,
            "fragment_interval": fragment_interval,
            "seed": seed,
            "media_type": media_type,
            "streaming_mode": streaming_mode,
            "parallel_infer": parallel_infer,
            "repetition_penalty": repetition_penalty
        }
        
        # Add reference audio if provided
        if ref_audio_path and os.path.exists(ref_audio_path):
            payload["ref_audio_path"] = ref_audio_path
            
        # Add prompt text if provided
        if prompt_text:
            payload["prompt_text"] = prompt_text
            
        try:
            response = self.session.post(
                f"{self.base_url}/tts",
                json=payload,
                timeout=120,
                stream=streaming_mode
            )
            
            if response.status_code == 200:
                if streaming_mode:
                    # Handle streaming response
                    audio_chunks = []
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            audio_chunks.append(chunk)
                    return b''.join(audio_chunks)
                else:
                    return response.content
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request Error: {e}")
            return None

def test_voice_cloning(args):
    """Main test function"""
    
    print("="*70)
    print("🎤 GPT-SoVITS Voice Cloning Test (Advanced API v2)")
    print("="*70)
    
    # Initialize client
    client = GPTSoVITSClient(host=args.host, port=args.port)
    
    # Check server health
    print("🔍 Checking API server status...")
    if not client.check_health():
        print(f"❌ API server not responding at {args.host}:{args.port}")
        print("💡 Tip: Make sure the GPT-SoVITS API server is running")
        return False
    print("✅ API server is running")
    
    # Set custom model weights if provided
    if args.gpt_weights:
        print(f"📦 Setting GPT weights: {args.gpt_weights}")
        if client.set_gpt_weights(args.gpt_weights):
            print("✅ GPT weights loaded")
        else:
            print("⚠️  Failed to load GPT weights")
    
    if args.sovits_weights:
        print(f"📦 Setting SoVITS weights: {args.sovits_weights}")
        if client.set_sovits_weights(args.sovits_weights):
            print("✅ SoVITS weights loaded")
        else:
            print("⚠️  Failed to load SoVITS weights")
    
    # Prepare parameters
    print("\n" + "="*70)
    print("📋 Configuration:")
    print(f"  Text: {args.text}")
    print(f"  Language: {args.text_lang}")
    if args.ref_audio:
        print(f"  Reference Audio: {args.ref_audio}")
    if args.prompt_text:
        print(f"  Prompt Text: {args.prompt_text}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Top-P: {args.top_p}")
    print(f"  Speed Factor: {args.speed}")
    print(f"  Streaming: {args.stream}")
    print(f"  Parallel Inference: {args.parallel}")
    print("="*70)
    
    # Perform voice cloning
    print("\n🔄 Generating audio...")
    start_time = time.time()
    
    audio_data = client.voice_clone(
        text=args.text,
        text_lang=args.text_lang,
        ref_audio_path=args.ref_audio,
        prompt_text=args.prompt_text,
        prompt_lang=args.prompt_lang,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        text_split_method=args.split_method,
        batch_size=args.batch_size,
        batch_threshold=args.batch_threshold,
        split_bucket=not args.no_split_bucket,
        speed_factor=args.speed,
        fragment_interval=args.fragment_interval,
        seed=args.seed,
        media_type=args.format,
        streaming_mode=args.stream,
        parallel_infer=args.parallel,
        repetition_penalty=args.repetition_penalty
    )
    
    generation_time = time.time() - start_time
    
    if audio_data:
        # Save output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(audio_data)
        
        file_size = len(audio_data)
        
        print("\n" + "="*70)
        print("✅ SUCCESS! Voice cloning completed!")
        print(f"🎧 Output saved: {output_path}")
        print(f"📊 File size: {file_size:,} bytes")
        print(f"⏱️  Generation time: {generation_time:.2f} seconds")
        print(f"🚀 Speed: {len(args.text) / generation_time:.1f} chars/sec")
        print("="*70)
        
        return True
    else:
        print("\n❌ Failed to generate audio")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Advanced GPT-SoVITS Voice Cloning Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Japanese text
  %(prog)s "こんにちは、今日はいい天気ですね" -o output.wav
  
  # With reference audio for voice cloning
  %(prog)s "新しいテキスト" --ref-audio reference.wav -o cloned.wav
  
  # With custom parameters for better quality
  %(prog)s "テキスト" --temperature 0.3 --top-k 5 --speed 0.9
  
  # Streaming mode for long text
  %(prog)s "長いテキスト..." --stream -o long_output.wav
        """
    )
    
    # Required arguments
    parser.add_argument('text', help='Text to synthesize')
    
    # Connection settings
    parser.add_argument('--host', default='127.0.0.1', help='API server host')
    parser.add_argument('--port', type=int, default=9885, help='API server port')
    
    # Model settings
    parser.add_argument('--gpt-weights', help='Path to GPT model weights')
    parser.add_argument('--sovits-weights', help='Path to SoVITS model weights')
    
    # Voice cloning settings
    parser.add_argument('--ref-audio', help='Reference audio for voice cloning')
    parser.add_argument('--prompt-text', help='Prompt text matching reference audio')
    parser.add_argument('--prompt-lang', default='ja', choices=['zh', 'ja', 'en'], 
                       help='Prompt language')
    
    # Text settings
    parser.add_argument('--text-lang', default='ja', choices=['zh', 'ja', 'en', 'auto', 'auto_yue'],
                       help='Text language')
    parser.add_argument('--split-method', default='cut5',
                       choices=['cut0', 'cut1', 'cut2', 'cut3', 'cut4', 'cut5'],
                       help='Text splitting method')
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, default=1.0, 
                       help='Sampling temperature (0.1-2.0)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Top-K sampling parameter')
    parser.add_argument('--top-p', type=float, default=1.0,
                       help='Top-P (nucleus) sampling parameter')
    parser.add_argument('--repetition-penalty', type=float, default=1.35,
                       help='Repetition penalty')
    
    # Performance settings
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--batch-threshold', type=float, default=0.75,
                       help='Batch threshold')
    parser.add_argument('--no-split-bucket', action='store_true',
                       help='Disable split bucket processing')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Speed factor (0.5-2.0)')
    parser.add_argument('--fragment-interval', type=float, default=0.3,
                       help='Fragment interval for streaming')
    parser.add_argument('--seed', type=int, default=-1,
                       help='Random seed (-1 for random)')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Enable parallel inference')
    parser.add_argument('--no-parallel', dest='parallel', action='store_false',
                       help='Disable parallel inference')
    parser.add_argument('--stream', action='store_true',
                       help='Enable streaming mode')
    
    # Output settings
    parser.add_argument('-o', '--output', default='output/test_clone.wav',
                       help='Output audio file path')
    parser.add_argument('--format', default='wav', choices=['wav', 'ogg', 'aac'],
                       help='Output audio format')
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.ref_audio and not os.path.exists(args.ref_audio):
        print(f"❌ Reference audio not found: {args.ref_audio}")
        return 1
    
    if args.temperature < 0.1 or args.temperature > 2.0:
        print("⚠️  Temperature should be between 0.1 and 2.0")
        args.temperature = max(0.1, min(2.0, args.temperature))
    
    if args.speed < 0.5 or args.speed > 2.0:
        print("⚠️  Speed should be between 0.5 and 2.0")
        args.speed = max(0.5, min(2.0, args.speed))
    
    # Run test
    success = test_voice_cloning(args)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())