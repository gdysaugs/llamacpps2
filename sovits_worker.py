#!/usr/bin/env python3
"""
SoVITS Worker - 単発チャンク音声生成用サブプロセス
成功したdirect_inference_separate.pyと同じTTSコードを使用
"""

import os
import sys
import argparse
import soundfile as sf
import json
from pathlib import Path

def setup_environment():
    """環境設定とパス設定"""
    base_path = Path(__file__).parent
    gpt_sovits_dir = base_path / "gpt_sovits_full"
    if not gpt_sovits_dir.exists():
        raise FileNotFoundError(f"GPT-SoVITS directory not found: {gpt_sovits_dir}")
    
    os.chdir(str(gpt_sovits_dir))
    sys.path.insert(0, str(gpt_sovits_dir))
    sys.path.insert(0, str(gpt_sovits_dir / "GPT_SoVITS"))
    
    return base_path, gpt_sovits_dir

def load_config(base_path):
    """設定ファイルを読み込み"""
    config_file = base_path / "tts_config.json"
    
    default_config = {
        "model_config_path": "gpt_sovits_full/GPT_SoVITS/configs/tts_infer.yaml",
        "ref_audio_path": "models/gpt_sovits/e_01_08_extended.wav",
        "prompt_text": "ああっ、気持ちいい。もっと、もっとして。",
        "output_dir": "output",
        "tts_params": {
            "top_k": 5,
            "top_p": 1,
            "temperature": 2.0,
            "text_split_method": "cut0",
            "speed_factor": 1.2,
            "batch_size": 4,
            "parallel_infer": True,
            "streaming_mode": False,
            "seed": 42
        }
    }
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict) and isinstance(config[key], dict):
                for sub_key, sub_value in value.items():
                    if sub_key not in config[key]:
                        config[key][sub_key] = sub_value
    else:
        config = default_config
    
    return config

def init_tts(config, base_path):
    """TTS推論パイプラインの初期化"""
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    
    model_config_path = base_path / config["model_config_path"]
    ref_audio_path = base_path / config["ref_audio_path"]
    
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    if not ref_audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")
    
    tts_config = TTS_Config(str(model_config_path))
    tts_pipeline = TTS(tts_config)
    
    return tts_pipeline, str(ref_audio_path)

def generate_single_chunk(text, output_file, config, tts_pipeline, ref_audio_path):
    """単一チャンクの音声生成（元のコードと同じ処理）"""
    try:
        # 句読点変換（元のコードと同じ）
        text = text.replace('。', '、')
        text = text.replace('！', '、')
        text = text.replace('？', '、')
        text = text.replace('.', '、')
        text = text.replace('!', '、')
        text = text.replace('?', '、')
        
        request = {
            "text": text,
            "text_lang": "ja",
            "ref_audio_path": ref_audio_path,
            "prompt_text": config["prompt_text"],
            "prompt_lang": "ja",
            "return_fragment": False,
            **config["tts_params"]
        }
        
        # TTS実行（成功版と同じgenerator処理）
        tts_generator = tts_pipeline.run(request)
        
        audio_chunks = []
        sample_rate = None
        
        for sr, chunk in tts_generator:
            if sample_rate is None:
                sample_rate = sr
            audio_chunks.append(chunk)
        
        if audio_chunks and sample_rate is not None:
            import numpy as np
            audio_data = np.concatenate(audio_chunks, axis=0)
            sf.write(output_file, audio_data, sample_rate)
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='SoVITS単発チャンク音声生成')
    parser.add_argument('text', help='生成するテキスト')
    parser.add_argument('output', help='出力ファイルパス')
    args = parser.parse_args()
    
    try:
        # 環境設定
        base_path, gpt_sovits_dir = setup_environment()
        
        # 設定読み込み
        config = load_config(base_path)
        
        # TTS初期化
        tts_pipeline, ref_audio_path = init_tts(config, base_path)
        
        # 音声生成
        success = generate_single_chunk(args.text, args.output, config, tts_pipeline, ref_audio_path)
        
        if success:
            print(f"✅ 音声生成完了: {args.output}")
            sys.exit(0)
        else:
            print(f"❌ 音声生成失敗")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()