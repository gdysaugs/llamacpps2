#!/usr/bin/env python3
"""
GPT-SoVITS ポータブル直接推論スクリプト
exe配布対応版 - 実行ファイルの場所を基準に動作
"""

import os
import sys
import soundfile as sf
import json
from pathlib import Path

def get_base_path():
    """実行ファイルまたはスクリプトの基準ディレクトリを取得"""
    if getattr(sys, 'frozen', False):
        # exe化された場合
        base_path = Path(sys.executable).parent
    else:
        # スクリプト実行の場合
        base_path = Path(__file__).parent
    return base_path

def setup_environment():
    """環境設定とパス設定"""
    base_path = get_base_path()
    
    # GPT-SoVITSディレクトリ設定
    gpt_sovits_dir = base_path / "gpt_sovits_full"
    if not gpt_sovits_dir.exists():
        raise FileNotFoundError(f"GPT-SoVITS directory not found: {gpt_sovits_dir}")
    
    # 作業ディレクトリを変更
    os.chdir(str(gpt_sovits_dir))
    
    # Pythonパス設定
    sys.path.insert(0, str(gpt_sovits_dir))
    sys.path.insert(0, str(gpt_sovits_dir / "GPT_SoVITS"))
    
    return base_path, gpt_sovits_dir

class PortableTTS:
    def __init__(self):
        self.base_path, self.gpt_sovits_dir = setup_environment()
        self.config = self.load_config()
        self.tts_pipeline = None
        
    def load_config(self):
        """設定ファイルを読み込み"""
        config_file = self.base_path / "tts_config.json"
        
        # デフォルト設定
        default_config = {
            "model_config_path": "GPT_SoVITS/configs/tts_infer.yaml",
            "ref_audio_path": "models/gpt_sovits/e_01_08_extended.wav", 
            "prompt_text": "ああっ、気持ちいい。もっと、もっとして。",
            "output_dir": "output",
            "tts_params": {
                "top_k": 5,
                "top_p": 1,
                "temperature": 1.5,
                "text_split_method": "cut0",
                "speed_factor": 1.2,
                "batch_size": 4,
                "parallel_infer": True,
                "streaming_mode": False,
                "seed": 42
            }
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"警告: 設定ファイル読み込みエラー: {e}")
                print("デフォルト設定を使用します")
        else:
            # 初回実行時に設定ファイルを作成
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            print(f"設定ファイルを作成しました: {config_file}")
        
        return default_config
    
    def init_tts(self):
        """TTS システムを初期化"""
        # 動的インポート（環境設定後）
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        
        # パスが相対パスの場合はgpt_sovits_dirからの相対、絶対パスの場合はそのまま
        model_config_path = self.config["model_config_path"]
        if model_config_path.startswith("gpt_sovits_full/"):
            # gpt_sovits_full/で始まる場合はbase_pathからの相対パス
            config_path = self.base_path / model_config_path
        else:
            # それ以外はgpt_sovits_dirからの相対パス
            config_path = self.gpt_sovits_dir / model_config_path
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        tts_config = TTS_Config(str(config_path))
        self.tts_pipeline = TTS(tts_config)
        print("TTS システム初期化完了")
        return self.tts_pipeline
    
    def generate_speech(self, text, output_filename=None):
        """音声生成"""
        if self.tts_pipeline is None:
            raise RuntimeError("TTS システムが初期化されていません")
        
        # パス設定
        ref_audio_path = self.base_path / self.config["ref_audio_path"]
        if not ref_audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")
        
        output_dir = self.base_path / self.config["output_dir"]
        output_dir.mkdir(exist_ok=True)
        
        if output_filename is None:
            import time
            timestamp = int(time.time())
            output_filename = f"generated_{timestamp}.wav"
        
        output_path = output_dir / output_filename
        
        # リクエスト作成
        request = {
            "text": text,
            "text_lang": "ja",
            "ref_audio_path": str(ref_audio_path),
            "prompt_text": self.config["prompt_text"],
            "prompt_lang": "ja",
            "return_fragment": False,
            **self.config["tts_params"]  # 設定ファイルからパラメータを取得
        }
        
        print(f"生成中: {text[:50]}...")
        print(f"出力先: {output_path}")
        
        # 音声生成
        tts_generator = self.tts_pipeline.run(request)
        
        # 音声データを結合
        audio_chunks = []
        sample_rate = None
        
        for sr, chunk in tts_generator:
            if sample_rate is None:
                sample_rate = sr
            audio_chunks.append(chunk)
        
        if audio_chunks:
            import numpy as np
            full_audio = np.concatenate(audio_chunks, axis=0)
            sf.write(str(output_path), full_audio, sample_rate)
            
            duration = len(full_audio) / sample_rate
            print(f"✅ 生成完了: {output_filename} ({duration:.2f}秒)")
            return str(output_path)
        else:
            print("❌ 音声生成失敗")
            return None
    
    def batch_generate(self, text_list, output_prefix="batch"):
        """バッチ音声生成"""
        results = []
        for i, text in enumerate(text_list, 1):
            output_filename = f"{output_prefix}_{i:03d}.wav"
            result = self.generate_speech(text, output_filename)
            results.append(result)
        return results

def main():
    """メイン関数"""
    try:
        print("GPT-SoVITS ポータブル直接推論システム")
        print("=" * 50)
        
        # TTS初期化
        portable_tts = PortableTTS()
        portable_tts.init_tts()
        
        # テスト用テキスト
        test_texts = [
            "んああああんっ、いっちゃう、いやだあああ、やめてえええ、",
            "ああっ、だめ、そんなところ触っちゃだめ、でも気持ちいい、もっと強く、"
        ]
        
        print("\nテスト音声生成開始...")
        results = portable_tts.batch_generate(test_texts, "portable_test")
        
        print(f"\n🎉 {len([r for r in results if r])}個のファイルが生成されました")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()