#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 推論ファイル (gpt_sovits_env専用)
GPU対応・v4モデル・日本語対応

使用方法:
source gpt_sovits_env/bin/activate
python gpt_sovits_inference_env.py "生成したいテキスト" output.wav

Requirements:
- gpt_sovits_env環境アクティベート必須
- models/gpt_sovits/gpt_sovits_model.ckpt
- models/gpt_sovits/pretrained_models/s2Gv4.pth
"""

import os
import sys
import argparse
import traceback
import logging
from pathlib import Path
import torch
import yaml
from typing import Optional

# プロジェクトルートディレクトリの設定
PROJECT_ROOT = Path(__file__).parent.absolute()
MODELS_ROOT = PROJECT_ROOT / "models/gpt_sovits"
PRETRAINED_ROOT = MODELS_ROOT / "pretrained_models"

# パスの設定 - ERes2NetV2の場所を追加
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PRETRAINED_ROOT))

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 必要な依存モジュールのimport
try:
    import numpy as np
    import torch
    import torchaudio
    import soundfile as sf
    from transformers import AutoModel, AutoTokenizer
    import librosa
    import json
    logger.info("✅ 基本ライブラリのインポート完了")
except ImportError as e:
    logger.error(f"基本ライブラリのインポートエラー: {e}")
    raise

# GPT-SoVITS modules import
try:
    from tools.i18n.i18n import I18nAuto
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names
    logger.info("✅ GPT-SoVITS コアモジュールのインポート完了")
except ImportError as e:
    logger.error(f"GPT-SoVITSモジュールのインポートエラー: {e}")
    raise

class GPTSoVITSInferenceEnv:
    """
    GPT-SoVITS推論クラス (gpt_sovits_env専用)
    """

    def __init__(self):
        """初期化"""
        self.project_root = PROJECT_ROOT
        self.models_root = MODELS_ROOT
        self.tts_pipeline = None
        self.tts_config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"🚀 GPT-SoVITS推論クラス初期化")
        logger.info(f"プロジェクトルート: {self.project_root}")
        logger.info(f"モデルディレクトリ: {self.models_root}")
        logger.info(f"使用デバイス: {self.device}")

        self._initialize()

    def _create_config(self):
        """設定の作成"""
        # 動的設定作成
        config_data = {
            "custom": {
                "device": self.device,
                "is_half": self.device == "cuda",
                "version": "v4",
                "bert_base_path": str(PRETRAINED_ROOT / "chinese-roberta-wwm-ext-large"),
                "cnhuhbert_base_path": str(PRETRAINED_ROOT / "chinese-hubert-base"),
                "t2s_weights_path": str(MODELS_ROOT / "gpt_sovits_model.ckpt"),
                "vits_weights_path": str(PRETRAINED_ROOT / "s2Gv4.pth")
            }
        }

        # 設定ファイルの作成
        config_path = self.project_root / "tts_infer_env.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"✅ 設定ファイル作成: {config_path}")
        return config_path

    def _initialize(self):
        """設定とモデルの初期化"""
        try:
            # 必要なモデルファイルの存在確認
            gpt_model_path = MODELS_ROOT / "gpt_sovits_model.ckpt"
            vits_model_path = PRETRAINED_ROOT / "s2Gv4.pth"

            if not gpt_model_path.exists():
                raise FileNotFoundError(f"GPTモデルが見つかりません: {gpt_model_path}")
            if not vits_model_path.exists():
                raise FileNotFoundError(f"VITSモデルが見つかりません: {vits_model_path}")

            logger.info(f"✅ GPTモデル確認: {gpt_model_path}")
            logger.info(f"✅ VITSモデル確認: {vits_model_path}")

            # 設定ファイルの作成
            config_path = self._create_config()

            # TTS設定とパイプラインの初期化
            self.tts_config = TTS_Config(str(config_path), config_name="custom")

            logger.info(f"デバイス: {self.tts_config.device}")
            logger.info(f"Half precision: {self.tts_config.is_half}")
            logger.info(f"モデルバージョン: {self.tts_config.version}")

            # TTSパイプラインの初期化
            logger.info("🔄 TTSパイプライン初期化中...")
            self.tts_pipeline = TTS(self.tts_config)
            logger.info("✅ GPT-SoVITS初期化完了")

        except Exception as e:
            logger.error(f"初期化エラー: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def generate_audio(self,
                      text: str,
                      ref_audio_path: str = None,
                      prompt_text: str = None,
                      text_lang: str = "ja",
                      prompt_lang: str = "ja",
                      output_path: str = "output/generated.wav",
                      **kwargs) -> str:
        """
        音声生成

        Args:
            text (str): 生成するテキスト
            ref_audio_path (str): リファレンス音声パス
            prompt_text (str): リファレンス音声のテキスト
            text_lang (str): テキストの言語
            prompt_lang (str): プロンプトの言語
            output_path (str): 出力ファイルパス
            **kwargs: その他のTTSパラメータ

        Returns:
            str: 生成された音声ファイルのパス
        """
        try:
            logger.info(f"🎤 音声生成開始: '{text[:50]}{'...' if len(text) > 50 else ''}'")

            # デフォルトリファレンス音声の設定
            if ref_audio_path is None:
                # 利用可能なリファレンス音声を探す
                ref_candidates = [
                    "hajimemashite_reference.wav",
                    "hajimemashite_reference.mp3",
                    "baka_reference.wav",
                    "test_reference.wav",
                    "reference.wav"
                ]

                ref_audio_path = None
                for candidate in ref_candidates:
                    candidate_path = MODELS_ROOT / candidate
                    if candidate_path.exists():
                        ref_audio_path = str(candidate_path)
                        break

                if ref_audio_path is None:
                    raise FileNotFoundError("利用可能なリファレンス音声が見つかりません")

            if prompt_text is None:
                prompt_text = "はじめまして、こんにちは"

            # リファレンス音声の自動変換
            ref_audio_path = self._ensure_wav_format(ref_audio_path)
            logger.info(f"📻 リファレンス音声: {ref_audio_path}")

            # 出力ディレクトリの作成
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # TTS実行パラメータの設定
            request_params = {
                "text": text,
                "text_lang": text_lang.lower(),
                "ref_audio_path": ref_audio_path,
                "prompt_text": prompt_text,
                "prompt_lang": prompt_lang.lower(),
                "top_k": kwargs.get("top_k", 15),
                "top_p": kwargs.get("top_p", 1.0),
                "temperature": kwargs.get("temperature", 1.0),
                "text_split_method": kwargs.get("text_split_method", "cut5"),
                "batch_size": kwargs.get("batch_size", 1),
                "speed_factor": kwargs.get("speed_factor", 1.0),
                "seed": kwargs.get("seed", -1),
                "parallel_infer": kwargs.get("parallel_infer", True),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.35),
                "sample_steps": kwargs.get("sample_steps", 32),
                "super_sampling": kwargs.get("super_sampling", False),
                "return_fragment": False
            }

            logger.info(f"📝 TTS パラメータ設定完了")

            # TTS実行
            logger.info("🔄 音声生成実行中...")
            tts_generator = self.tts_pipeline.run(request_params)

            # 音声データの取得
            sr, audio_data = next(tts_generator)

            # WAV形式で保存
            sf.write(str(output_path), audio_data, sr)

            # ファイルサイズと時間の計算
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            duration = len(audio_data) / sr

            logger.info(f"✅ 音声生成成功: {output_path}")
            logger.info(f"📊 ファイルサイズ: {file_size:.2f}MB, 再生時間: {duration:.2f}秒")

            return str(output_path)

        except Exception as e:
            logger.error(f"音声生成エラー: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _ensure_wav_format(self, audio_path: str) -> str:
        """
        音声ファイルをWAV形式に変換（必要に応じて）
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            logger.warning(f"音声ファイルが存在しません: {audio_path}")
            return str(audio_path)

        # 既にWAVファイルの場合はそのまま返す
        if audio_path.suffix.lower() == ".wav":
            return str(audio_path)

        # WAVファイルパスの生成
        wav_path = audio_path.with_suffix(".wav")

        # 既に変換済みの場合はそれを使用
        if wav_path.exists():
            logger.info(f"✅ 変換済みWAVファイルを使用: {wav_path}")
            return str(wav_path)

        # ffmpegを使用してWAVに変換
        try:
            import subprocess
            logger.info(f"🔄 {audio_path.suffix}をWAVに変換中: {audio_path} -> {wav_path}")

            cmd = [
                "ffmpeg", "-i", str(audio_path),
                "-ar", "22050", "-ac", "1", "-y", str(wav_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"✅ WAV変換完了: {wav_path}")
                return str(wav_path)
            else:
                logger.warning(f"WAV変換失敗、元ファイルを使用: {audio_path}")
                return str(audio_path)

        except Exception as e:
            logger.warning(f"WAV変換エラー、元ファイルを使用: {str(e)}")
            return str(audio_path)


def main():
    """コマンドライン実行用メイン関数"""
    parser = argparse.ArgumentParser(description="GPT-SoVITS 推論実行 (env専用)")
    parser.add_argument("text", help="生成するテキスト")
    parser.add_argument("output", help="出力ファイル名")
    parser.add_argument("--ref_audio", help="リファレンス音声パス")
    parser.add_argument("--prompt_text", help="リファレンス音声のテキスト")
    parser.add_argument("--text_lang", default="ja", help="テキストの言語 (デフォルト: ja)")
    parser.add_argument("--prompt_lang", default="ja", help="プロンプトの言語 (デフォルト: ja)")

    args = parser.parse_args()

    try:
        # 推論クラスの初期化
        logger.info("🚀 GPT-SoVITS推論開始 (env版)")
        inference = GPTSoVITSInferenceEnv()

        # 音声生成
        result_path = inference.generate_audio(
            text=args.text,
            ref_audio_path=args.ref_audio,
            prompt_text=args.prompt_text,
            text_lang=args.text_lang,
            prompt_lang=args.prompt_lang,
            output_path=args.output
        )

        print(f"✅ 音声生成完了: {result_path}")

    except Exception as e:
        logger.error(f"実行エラー: {str(e)}")
        sys.exit(1)


# Gradio/Wav2Lip統合用関数
def generate_audio_for_wav2lip(text: str,
                              output_filename: str = "generated_audio.wav",
                              **kwargs) -> Optional[str]:
    """
    Wav2Lip統合用の音声生成関数

    Args:
        text (str): 生成するテキスト
        output_filename (str): 出力ファイル名
        **kwargs: その他のパラメータ

    Returns:
        Optional[str]: 生成された音声ファイルのフルパス（失敗時はNone）
    """
    try:
        inference = GPTSoVITSInferenceEnv()
        output_path = PROJECT_ROOT / "output" / output_filename

        result_path = inference.generate_audio(
            text=text,
            output_path=str(output_path),
            **kwargs
        )

        return result_path
    except Exception as e:
        logger.error(f"Wav2Lip統合用音声生成エラー: {str(e)}")
        return None


if __name__ == "__main__":
    main()