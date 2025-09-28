#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 推論ファイル (api_v2ベース)
GPU対応・v4モデル・日本語対応・Wav2Lip統合用

使用方法:
python gpt_sovits_inference_apiv2.py "生成したいテキスト" output.wav

または
from gpt_sovits_inference_apiv2 import GPTSoVITSInference

Requirements:
- gpt_sovits_env環境アクティベート必須
- models/gpt_sovits/gpt_sovits_model.ckpt
- models/gpt_sovits/pretrained_models/s2Gv4.pth
"""

import os
import sys
import argparse
import traceback
from pathlib import Path
import torch
import logging
from typing import Optional
import json

# プロジェクトルートディレクトリの設定
PROJECT_ROOT = Path(__file__).parent.absolute()
GPT_SOVITS_ROOT = PROJECT_ROOT / "gpt_sovits_full"

# パスの追加
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(GPT_SOVITS_ROOT))
sys.path.insert(0, str(GPT_SOVITS_ROOT / "GPT_SoVITS"))

# GPT-SoVITSのimport
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPTSoVITSInference:
    """
    GPT-SoVITS推論クラス
    api_v2.pyをベースに、シンプルで効果的な音声生成を提供
    """

    def __init__(self, config_name="custom"):
        """
        初期化

        Args:
            config_name (str): tts_infer.yamlの設定名 (デフォルト: "custom")
        """
        self.project_root = PROJECT_ROOT
        self.config_name = config_name
        self.tts_pipeline = None
        self.tts_config = None

        # 設定の読み込みと初期化
        self._initialize()

    def _initialize(self):
        """設定とモデルの初期化"""
        try:
            # 設定ファイルのパス
            config_path = self.project_root / "gpt_sovits_full/GPT_SoVITS/configs/tts_infer.yaml"

            if not config_path.exists():
                raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

            logger.info(f"設定ファイルを読み込み中: {config_path}")
            logger.info(f"使用する設定: {self.config_name}")

            # TTS設定とパイプラインの初期化
            self.tts_config = TTS_Config(str(config_path), config_name=self.config_name)

            # GPU使用可能性チェック
            if self.tts_config.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA利用不可。CPUにフォールバック")
                self.tts_config.device = "cpu"
                self.tts_config.is_half = False

            logger.info(f"デバイス: {self.tts_config.device}")
            logger.info(f"Half precision: {self.tts_config.is_half}")
            logger.info(f"モデルバージョン: {self.tts_config.version}")

            # TTSパイプラインの初期化
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
            ref_audio_path (str): リファレンス音声パス（Noneの場合デフォルト使用）
            prompt_text (str): リファレンス音声のテキスト（Noneの場合デフォルト使用）
            text_lang (str): テキストの言語（デフォルト: "ja"）
            prompt_lang (str): プロンプトの言語（デフォルト: "ja"）
            output_path (str): 出力ファイルパス
            **kwargs: その他のTTSパラメータ

        Returns:
            str: 生成された音声ファイルのパス
        """
        try:
            logger.info(f"🎤 音声生成開始: '{text[:50]}{'...' if len(text) > 50 else ''}'")

            # デフォルトリファレンス音声の設定
            if ref_audio_path is None:
                ref_audio_path = str(self.project_root / "models/gpt_sovits/hajimemashite_reference.wav")

            if prompt_text is None:
                prompt_text = "はじめまして、こんにちは"

            # リファレンス音声の存在確認と自動変換
            ref_audio_path = self._ensure_wav_format(ref_audio_path)

            if not Path(ref_audio_path).exists():
                raise FileNotFoundError(f"リファレンス音声が見つかりません: {ref_audio_path}")

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

            logger.info(f"📝 パラメータ: {json.dumps({k: v for k, v in request_params.items() if k != 'text'}, indent=2, ensure_ascii=False)}")

            # TTS実行
            logger.info("🔄 音声生成実行中...")
            tts_generator = self.tts_pipeline.run(request_params)

            # 音声データの取得
            sr, audio_data = next(tts_generator)

            # WAV形式で保存
            import soundfile as sf
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

        Args:
            audio_path (str): 元の音声ファイルパス

        Returns:
            str: WAVファイルのパス
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
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
            logger.info(f"🔄 .{audio_path.suffix}をWAVに変換中: {audio_path} -> {wav_path}")

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

    def set_reference_audio(self, ref_audio_path: str):
        """
        デフォルトリファレンス音声の設定

        Args:
            ref_audio_path (str): リファレンス音声のパス
        """
        try:
            ref_audio_path = self._ensure_wav_format(ref_audio_path)
            self.tts_pipeline.set_ref_audio(ref_audio_path)
            logger.info(f"✅ リファレンス音声設定完了: {ref_audio_path}")
        except Exception as e:
            logger.error(f"リファレンス音声設定エラー: {str(e)}")
            raise


def main():
    """コマンドライン実行用メイン関数"""
    parser = argparse.ArgumentParser(description="GPT-SoVITS 推論実行")
    parser.add_argument("text", help="生成するテキスト")
    parser.add_argument("output", help="出力ファイル名")
    parser.add_argument("--ref_audio", help="リファレンス音声パス")
    parser.add_argument("--prompt_text", help="リファレンス音声のテキスト")
    parser.add_argument("--text_lang", default="ja", help="テキストの言語 (デフォルト: ja)")
    parser.add_argument("--prompt_lang", default="ja", help="プロンプトの言語 (デフォルト: ja)")
    parser.add_argument("--config", default="custom", help="設定名 (デフォルト: custom)")

    args = parser.parse_args()

    try:
        # 推論クラスの初期化
        logger.info("🚀 GPT-SoVITS推論開始")
        inference = GPTSoVITSInference(config_name=args.config)

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


# Gradio統合用関数
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
        inference = GPTSoVITSInference()
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