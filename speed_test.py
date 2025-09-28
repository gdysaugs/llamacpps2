#!/usr/bin/env python3
"""
GPT-SoVITS 速度比較テスト
batch_size=1 vs batch_size=4
"""

import os
import sys
import soundfile as sf
import time
import numpy as np

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

def generate_speech_batch_test(tts_pipeline, text, batch_size, output_path):
    """音声生成（batch_sizeテスト用）"""
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
        "batch_size": batch_size,
        "parallel_infer": True,
        "streaming_mode": False,
        "return_fragment": False
    }
    
    print(f"生成中 (batch_size={batch_size}): {text[:30]}...")
    start_time = time.time()
    
    tts_generator = tts_pipeline.run(request)
    
    # 全ての音声チャンクを結合
    audio_chunks = []
    sample_rate = None
    
    for sr, chunk in tts_generator:
        if sample_rate is None:
            sample_rate = sr
        audio_chunks.append(chunk)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks, axis=0)
        sf.write(output_path, full_audio, sample_rate)
        duration = len(full_audio) / sample_rate
        print(f"✅ 完了: {output_path}")
        print(f"   処理時間: {processing_time:.2f}秒")
        print(f"   音声長: {duration:.2f}秒")
        print(f"   RTF: {processing_time/duration:.2f} (低いほど高速)")
        return processing_time, duration
    else:
        print("❌ エラー: 音声生成失敗")
        return None, None

def main():
    """メイン関数"""
    print("GPT-SoVITS 速度比較テスト開始...")
    tts_pipeline = init_tts()
    print("初期化完了！\n")
    
    # テスト用テキスト
    test_text = "んああああんっ、いっちゃう、いやだあああ、やめてえええ、キモイ、キモイっ、死ね、出すなあああ、もう限界、気持ちよすぎる"
    
    output_dir = "/home/adama/wav2lip-project/output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== batch_size=1 テスト ===")
    time1, duration1 = generate_speech_batch_test(
        tts_pipeline, test_text, 1, f"{output_dir}/batch1_test.wav"
    )
    
    print("\n=== batch_size=4 テスト ===")
    time4, duration4 = generate_speech_batch_test(
        tts_pipeline, test_text, 4, f"{output_dir}/batch4_test.wav"
    )
    
    if time1 and time4:
        print("\n" + "="*50)
        print("速度比較結果:")
        print(f"batch_size=1: {time1:.2f}秒 (RTF: {time1/duration1:.2f})")
        print(f"batch_size=4: {time4:.2f}秒 (RTF: {time4/duration4:.2f})")
        
        if time4 < time1:
            speedup = time1 / time4
            print(f"🚀 batch_size=4 が {speedup:.2f}x 高速!")
        else:
            slowdown = time4 / time1
            print(f"⚠️ batch_size=4 が {slowdown:.2f}x 低速")
    
    print("\nテスト完了!")

if __name__ == "__main__":
    main()