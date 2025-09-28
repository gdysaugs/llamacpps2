#!/usr/bin/env python3
"""
GPT-SoVITS 分割推論スクリプト
長文を短文に分割して個別推論し、最後に結合
"""

import os
import sys
import soundfile as sf
import json
import numpy as np
import re
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

class ChunkedTTS:
    def __init__(self):
        self.base_path, self.gpt_sovits_dir = setup_environment()
        self.config = self.load_config()
        self.tts_pipeline = None
        
    def load_config(self):
        """設定ファイルを読み込み"""
        config_file = self.base_path / "tts_config.json"
        
        default_config = {
            "model_config_path": "gpt_sovits_full/GPT_SoVITS/configs/tts_infer.yaml",
            "ref_audio_path": "models/gpt_sovits/e_01_08_extended.wav", 
            "prompt_text": "ああっ、気持ちいい。もっと、もっとして。",
            "output_dir": "output",
            "chunk_length": 50,  # 分割する最大文字数
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
        
        return default_config
    
    def init_tts(self):
        """TTS システムを初期化"""
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        
        model_config_path = self.config["model_config_path"]
        if model_config_path.startswith("gpt_sovits_full/"):
            config_path = self.base_path / model_config_path
        else:
            config_path = self.gpt_sovits_dir / model_config_path
            
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        tts_config = TTS_Config(str(config_path))
        self.tts_pipeline = TTS(tts_config)
        print("TTS システム初期化完了")
        return self.tts_pipeline
    
    def split_text(self, text, max_length=50):
        """テキストを適切な区切りで分割"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # 句読点で分割を試みる
        sentences = re.split(r'([。、！？…])', text)
        
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
            else:
                sentence = sentences[i]
            
            if len(current_chunk + sentence) <= max_length:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # まだ長すぎる場合は強制分割
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                final_chunks.append(chunk)
            else:
                # 強制的に文字数で分割
                for i in range(0, len(chunk), max_length):
                    final_chunks.append(chunk[i:i + max_length])
        
        return final_chunks
    
    def generate_chunk_speech(self, text_chunk, chunk_id):
        """1つのチャンクの音声を生成"""
        ref_audio_path = self.base_path / self.config["ref_audio_path"]
        
        request = {
            "text": text_chunk,
            "text_lang": "ja",
            "ref_audio_path": str(ref_audio_path),
            "prompt_text": self.config["prompt_text"],
            "prompt_lang": "ja",
            "return_fragment": False,
            **self.config["tts_params"]
        }
        
        print(f"チャンク {chunk_id}: {text_chunk[:30]}... ({len(text_chunk)}文字)")
        
        tts_generator = self.tts_pipeline.run(request)
        
        audio_chunks = []
        sample_rate = None
        
        for sr, chunk in tts_generator:
            if sample_rate is None:
                sample_rate = sr
            audio_chunks.append(chunk)
        
        if audio_chunks:
            return np.concatenate(audio_chunks, axis=0), sample_rate
        else:
            print(f"❌ チャンク {chunk_id} 生成失敗")
            return None, None
    
    def generate_chunked_speech(self, text, output_filename=None):
        """分割推論で音声生成"""
        if self.tts_pipeline is None:
            raise RuntimeError("TTS システムが初期化されていません")
        
        # テキスト分割
        chunk_length = self.config.get("chunk_length", 50)
        text_chunks = self.split_text(text, chunk_length)
        
        print(f"テキストを {len(text_chunks)} 個のチャンクに分割")
        for i, chunk in enumerate(text_chunks, 1):
            print(f"  チャンク{i}: {chunk}")
        
        # 各チャンクを生成
        all_audio_data = []
        sample_rate = None
        
        for i, text_chunk in enumerate(text_chunks, 1):
            audio_data, sr = self.generate_chunk_speech(text_chunk, i)
            
            if audio_data is not None:
                if sample_rate is None:
                    sample_rate = sr
                
                # データ型とサンプルレート統一
                if sr != sample_rate:
                    print(f"警告: チャンク {i} のサンプルレートが異なります ({sr} vs {sample_rate})")
                    continue
                
                # オーディオデータを32bit floatに正規化
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # 振幅を正規化（クリッピング防止）
                max_val = np.abs(audio_data).max()
                if max_val > 0.95:  # クリッピング防止
                    audio_data = audio_data * (0.95 / max_val)
                
                all_audio_data.append(audio_data)
                
                # チャンク間に短い無音を挿入（自然な間）
                if i < len(text_chunks):
                    silence_samples = int(0.3 * sample_rate)  # 0.3秒に延長
                    silence = np.zeros(silence_samples, dtype=np.float32)
                    all_audio_data.append(silence)
        
        if not all_audio_data:
            print("❌ 全チャンク生成失敗")
            return None
        
        # 全オーディオを結合
        full_audio = np.concatenate(all_audio_data, axis=0)
        
        # 出力
        output_dir = self.base_path / self.config["output_dir"]
        output_dir.mkdir(exist_ok=True)
        
        if output_filename is None:
            import time
            timestamp = int(time.time())
            output_filename = f"chunked_{timestamp}.wav"
        
        output_path = output_dir / output_filename
        sf.write(str(output_path), full_audio, sample_rate)
        
        duration = len(full_audio) / sample_rate
        print(f"✅ 分割推論完了: {output_filename} ({duration:.2f}秒)")
        print(f"   総チャンク数: {len(text_chunks)}")
        return str(output_path)

def main():
    """メイン関数"""
    try:
        print("GPT-SoVITS 分割推論システム")
        print("=" * 50)
        
        chunked_tts = ChunkedTTS()
        chunked_tts.init_tts()
        
        # 超長文テスト
        ultra_long_text = """んああああんっ、気持ちいい、もっと激しくして、そこ、そこがいいの、だめ、いっちゃう、いっちゃうから、やめて、でも止めないで、もっと奥まで、ああんっ、感じちゃう、こんなに濡れてるの恥ずかしい、でも気持ちよくて止められない、もっと、もっと強く、激しく、ああっ、だめ、限界、いく、いくー、はあ、はあ、息が荒くなって、心臓がドキドキして、体が熱くなって、もうだめ、こんなの初めて、こんなに気持ちいいの初めて、もっと触って、もっと愛して、全部感じたい、全部あなたのものになりたい、ああんっ、またきちゃう、何度でもいっちゃう、止まらない、この快感止まらない、もう虜になっちゃった"""
        
        print(f"\n長文テスト開始... ({len(ultra_long_text)}文字)")
        result = chunked_tts.generate_chunked_speech(ultra_long_text, "chunked_long_test.wav")
        
        if result:
            print(f"🎉 分割推論成功!")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()