#!/usr/bin/env python3
"""
GPT-SoVITS 単発音声生成スクリプト（サブプロセス用）
コマンドライン引数でテキストと出力ファイル名を受け取り、単独で音声生成
"""

import os
import sys
import json
import argparse
from pathlib import Path

def get_base_path():
    """実行ファイルまたはスクリプトの基準ディレクトリを取得"""
    if getattr(sys, 'frozen', False):
        base_path = Path(sys.executable).parent
    else:
        base_path = Path(__file__).parent
    return base_path

def setup_environment():
    """環境設定とパス設定"""
    base_path = get_base_path()
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
        # デフォルト値とマージ
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
    try:
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        
        # 設定ファイルの絶対パス化
        model_config_path = base_path / config["model_config_path"]
        ref_audio_path = base_path / config["ref_audio_path"]
        
        if not model_config_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_config_path}")
        if not ref_audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")
        
        # TTS設定の作成
        gpt_path = None
        sovits_path = None
        device = "cuda"
        is_half = False
        
        tts_config = TTS_Config(str(model_config_path))
        tts_pipeline = TTS(tts_config)
        
        return tts_pipeline, str(ref_audio_path)
        
    except Exception as e:
        print(f"❌ TTS初期化エラー: {e}")
        raise

def generate_single_audio(text, output_file, config, tts_pipeline, ref_audio_path):
    """単発音声生成"""
    try:
        # TTSパラメータ取得
        tts_params = config["tts_params"]
        prompt_text = config["prompt_text"]
        
        print(f"テキスト: {text}")
        
        # TTSパイプライン実行（元のコードと同じAPIを使用）
        request = {
            "text": text,
            "text_lang": "ja",
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text,
            "prompt_lang": "ja",
            "return_fragment": False,
            **tts_params
        }
        
        sr, audio_data = tts_pipeline.run(request)
        
        if audio_data is not None:
            # 音声データを保存
            import soundfile as sf
            sf.write(output_file, audio_data, sr)
            return True
        else:
            print(f"❌ 音声生成失敗: {text[:20]}...")
            return False
            
    except Exception as e:
        print(f"❌ 音声生成エラー: {e}")
        return False

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='GPT-SoVITS 単発音声生成')
    parser.add_argument('text', help='生成するテキスト')
    parser.add_argument('output', help='出力ファイル名')
    args = parser.parse_args()
    
    try:
        # 環境設定
        base_path, gpt_sovits_dir = setup_environment()
        
        # 設定読み込み
        config = load_config(base_path)
        
        # TTS初期化
        tts_pipeline, ref_audio_path = init_tts(config, base_path)
        
        # 音声生成
        success = generate_single_audio(args.text, args.output, config, tts_pipeline, ref_audio_path)
        
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