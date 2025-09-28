#!/usr/bin/env python3
"""
GPT-SoVITS 直接推論スクリプト
APIを使わずに推論関数を直接呼び出し
"""

import os
import sys
import soundfile as sf

# 作業ディレクトリを変更（重要）
os.chdir("/home/adama/wav2lip-project/gpt_sovits_full")

# パス設定
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(f"{now_dir}/GPT_SoVITS")

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

def init_tts():
    """TTS システムを初期化"""
    config_path = "/home/adama/wav2lip-project/gpt_sovits_full/GPT_SoVITS/configs/tts_infer.yaml"
    tts_config = TTS_Config(config_path)
    tts_pipeline = TTS(tts_config)
    return tts_pipeline

def generate_speech(tts_pipeline, text, output_path):
    """音声生成（最適化パラメータ）"""
    request = {
        "text": text,
        "text_lang": "ja",
        "ref_audio_path": "/home/adama/wav2lip-project/models/gpt_sovits/e_01_08_extended.wav",
        "prompt_text": "ああっ、気持ちいい。もっと、もっとして。",
        "prompt_lang": "ja",
        "top_k": 5,
        "top_p": 1,
        "temperature": 1.5,
        "text_split_method": "cut0",
        "speed_factor": 1.2,
        "batch_size": 4,
        "parallel_infer": True,
        "streaming_mode": False,
        "return_fragment": False
    }
    
    print(f"生成中: {text[:30]}...")
    tts_generator = tts_pipeline.run(request)
    
    # 全ての音声チャンクを結合
    audio_chunks = []
    sample_rate = None
    
    for sr, chunk in tts_generator:
        if sample_rate is None:
            sample_rate = sr
        audio_chunks.append(chunk)
    
    if audio_chunks:
        import numpy as np
        full_audio = np.concatenate(audio_chunks, axis=0)
        sf.write(output_path, full_audio, sample_rate)
        print(f"保存完了: {output_path} ({len(full_audio)/sample_rate:.2f}秒)")
        return True
    else:
        print("エラー: 音声生成失敗")
        return False

def main():
    """メイン関数"""
    print("GPT-SoVITS 直接推論システム初期化中...")
    tts_pipeline = init_tts()
    print("初期化完了！")
    
    # テスト用テキスト
    test_texts = [
        "んああああんっ、いっちゃう、いやだあああ、やめてえええ、",
        "ああっ、だめ、そんなところ触っちゃだめ、でも気持ちいい、もっと強く、"
    ]
    
    output_dir = "/home/adama/wav2lip-project/output"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, text in enumerate(test_texts, 1):
        output_path = f"{output_dir}/direct_inference_{i}.wav"
        success = generate_speech(tts_pipeline, text, output_path)
        if success:
            print(f"✅ テスト {i} 成功")
        else:
            print(f"❌ テスト {i} 失敗")

if __name__ == "__main__":
    main()