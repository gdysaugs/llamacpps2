#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS シンプルCLI (v4モード専用)
ボイスクローンテスト用

使用方法:
source gpt_sovits_env/bin/activate
python gpt_sovits_simple_cli.py ref_audio.wav "生成したいテキスト" output.wav

例:
python gpt_sovits_simple_cli.py models/gpt_sovits/e_01_08_reference.wav "おはようございます、テストです" output/voice_clone_test.wav

注意: v4モードではリファレンステキストは不要です（target_textがprompt_textとして使用されます）
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import soundfile as sf
import traceback
import subprocess
import librosa

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.absolute()

# パス設定 (Docker対応)
if os.path.exists("/app/gpt_sovits_full"):
    # Docker環境
    gpt_sovits_root = Path("/app/gpt_sovits_full")
    sys.path.insert(0, str(gpt_sovits_root))
    sys.path.insert(0, str(gpt_sovits_root / "GPT_SoVITS"))
    sys.path.insert(0, str(gpt_sovits_root / "GPT_SoVITS" / "eres2net"))
else:
    # ローカル環境
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "gpt_sovits_full"))
    sys.path.insert(0, str(PROJECT_ROOT / "gpt_sovits_full" / "GPT_SoVITS"))
    sys.path.insert(0, str(PROJECT_ROOT / "gpt_sovits_full" / "GPT_SoVITS" / "eres2net"))

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # GPT-SoVITSのimport
    from tools.i18n.i18n import I18nAuto
    from TTS_infer_pack.TTS import TTS, TTS_Config
    logger.info("✅ GPT-SoVITS モジュール読み込み完了")
except ImportError as e:
    logger.error(f"❌ GPT-SoVITS モジュール読み込みエラー: {e}")
    logger.error(f"Python パス: {sys.path}")
    logger.error(f"現在の作業ディレクトリ: {os.getcwd()}")
    sys.exit(1)

class SimpleGPTSoVITSCLI:
    """シンプルなGPT-SoVITS CLIクラス"""

    def __init__(self):
        self.project_root = PROJECT_ROOT

        # Docker対応のmodelsパス設定
        if os.path.exists("/app/models/gpt_sovits"):
            # Docker環境
            self.models_root = Path("/app/models/gpt_sovits")
        else:
            # ローカル環境
            self.models_root = self.project_root / "models/gpt_sovits"

        self.tts_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 GPT-SoVITS CLI 初期化 (デバイス: {self.device})")
        logger.info(f"📁 モデルパス: {self.models_root}")

    def preprocess_audio(self, audio_path: str) -> str:
        """
        音声ファイルの前処理：
        1. WAV以外の形式をWAVに変換
        2. 3秒未満の場合は3秒に延長

        Args:
            audio_path: 元の音声ファイルパス

        Returns:
            str: 処理済みの音声ファイルパス
        """
        try:
            audio_path = Path(audio_path)

            # 出力ディレクトリを作成
            temp_dir = self.project_root / "temp_audio"
            temp_dir.mkdir(exist_ok=True)

            # ファイル拡張子をチェック
            if audio_path.suffix.lower() != '.wav':
                logger.info(f"📄 音声形式変換中: {audio_path.suffix} -> WAV")
                # 安全なファイル名を生成（日本語文字を避ける）
                import hashlib
                import time
                safe_name = f"audio_{int(time.time())}_{hashlib.md5(str(audio_path).encode()).hexdigest()[:8]}"
                temp_wav = temp_dir / f"{safe_name}_converted.wav"
                cmd = [
                    'ffmpeg', '-i', str(audio_path),
                    '-ar', '44100', '-ac', '1',
                    str(temp_wav), '-y'
                ]
                subprocess.run(cmd, capture_output=True, check=True)
                audio_path = temp_wav
                logger.info(f"✅ WAV変換完了: {temp_wav}")

            # 音声の長さを確認
            duration = librosa.get_duration(path=str(audio_path))
            logger.info(f"🎵 音声長さ: {duration:.2f}秒")

            if duration < 3.0:
                logger.info(f"⏱ 音声が短いため3秒に延長中...")
                # 3秒に延長（安全なファイル名を使用）
                if 'safe_name' not in locals():
                    import hashlib
                    import time
                    safe_name = f"audio_{int(time.time())}_{hashlib.md5(str(audio_path).encode()).hexdigest()[:8]}"
                extended_wav = temp_dir / f"{safe_name}_extended.wav"
                # 必要なサンプル数を計算 (44100 Hz * 2秒分を追加)
                pad_samples = int(44100 * (3.0 - duration))
                cmd = [
                    'ffmpeg', '-i', str(audio_path),
                    '-filter:a', f'apad=pad_len={pad_samples}',
                    str(extended_wav), '-y'
                ]
                subprocess.run(cmd, capture_output=True, check=True)
                audio_path = extended_wav
                logger.info(f"✅ 音声延長完了: {extended_wav} (3.00秒)")

            return str(audio_path)

        except Exception as e:
            logger.error(f"❌ 音声前処理エラー: {str(e)}")
            logger.error(traceback.format_exc())
            return str(audio_path)  # エラーの場合は元のパスを返す

    def initialize_models(self):
        """モデルの初期化"""
        try:
            # Docker環境でディレクトリ作成
            if os.path.exists("/app/gpt_sovits_full"):
                gsv_dir = Path("/app/gpt_sovits_full/GPT_SoVITS/pretrained_models/gsv-v4-pretrained")
                gsv_dir.mkdir(parents=True, exist_ok=True)
                # s2Gv4.pthファイルを正しい場所にコピー/リンク
                s2g_source = Path("/app/gpt_sovits_full/GPT_SoVITS/pretrained_models/s2Gv4.pth")
                s2g_target = gsv_dir / "s2Gv4.pth"
                if s2g_source.exists() and not s2g_target.exists():
                    import shutil
                    shutil.copy2(s2g_source, s2g_target)
                    logger.info(f"✅ モデルファイルコピー: {s2g_source} -> {s2g_target}")

            # 設定ファイルを環境内に作成
            # Docker対応のプリトレーニング済みモデルパス設定
            if os.path.exists("/app/gpt_sovits_full"):
                # Docker環境での絶対パス
                bert_path = "/app/gpt_sovits_full/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
                hubert_path = "/app/gpt_sovits_full/GPT_SoVITS/pretrained_models/chinese-hubert-base"
                # デフォルトパスと同じディレクトリ構造に変更
                vits_path = "/app/gpt_sovits_full/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
            else:
                # ローカル環境での相対パス
                bert_path = str(self.models_root / "pretrained_models/chinese-roberta-wwm-ext-large")
                hubert_path = str(self.models_root / "pretrained_models/chinese-hubert-base")
                vits_path = str(self.models_root / "pretrained_models/gsv-v4-pretrained/s2Gv4.pth")

            config_data = {
                "custom": {
                    "device": self.device,
                    "is_half": self.device == "cuda",
                    "version": "v4",
                    "bert_base_path": bert_path,
                    "cnhuhbert_base_path": hubert_path,
                    "t2s_weights_path": str(self.models_root / "gpt_sovits_model.ckpt"),
                    "vits_weights_path": vits_path
                }
            }

            # 設定ファイルの作成
            import yaml
            config_path = self.project_root / "tts_infer_env.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"✅ 設定ファイル作成: {config_path}")
            logger.info("🔄 TTS設定とパイプラインを初期化中...")

            # TTS設定とパイプライン
            self.tts_config = TTS_Config(str(config_path))
            self.tts_pipeline = TTS(self.tts_config)

            logger.info("✅ モデル初期化完了")
            return True

        except Exception as e:
            logger.error(f"❌ モデル初期化エラー: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def generate_voice_clone(self,
                           ref_text: str,
                           ref_audio_path: str,
                           target_text: str,
                           output_path: str,
                           speed_factor: float = 1.0) -> bool:
        """
        ボイスクローン音声生成

        Args:
            ref_text: リファレンステキスト
            ref_audio_path: リファレンス音声ファイル
            target_text: 生成したいテキスト
            output_path: 出力ファイルパス

        Returns:
            bool: 成功したかどうか
        """
        try:
            logger.info(f"🎤 ボイスクローン開始")
            logger.info(f"📝 リファレンステキスト: {ref_text}")
            logger.info(f"📻 リファレンス音声: {ref_audio_path}")
            logger.info(f"🎯 生成テキスト: {target_text}")

            # 音声ファイルの前処理
            processed_audio_path = self.preprocess_audio(ref_audio_path)

            # ファイル存在確認
            ref_audio_path = Path(processed_audio_path)
            if not ref_audio_path.exists():
                logger.error(f"❌ リファレンス音声が見つかりません: {ref_audio_path}")
                return False

            # 出力ディレクトリをoutputフォルダに強制設定
            output_dir = self.project_root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 出力ファイル名のみ取得してoutputディレクトリと結合
            output_filename = Path(output_path).name
            output_path = output_dir / output_filename

            # v4モード用：prompt_textをtarget_textと同じに設定
            prompt_text = target_text

            # TTSパラメータの設定（感情的・自然な音声用）
            request_params = {
                "text": target_text,
                "text_lang": "ja",
                "ref_audio_path": str(ref_audio_path),
                "prompt_text": prompt_text,
                "prompt_lang": "ja",
                "top_k": 5,  # 少数精鋭で感情的な候補
                "top_p": 0.75,  # 多様性を高めて感情表現豊かに
                "temperature": 2.0,  # 感情表現を非常に豊かに
                "text_split_method": "cut5",
                "batch_size": 1,
                "speed_factor": speed_factor,  # 動的に速度を変更
                "seed": -1,
                "parallel_infer": True,
                "repetition_penalty": 1.2,  # 繰り返しを少し許容
                "sample_steps": 48,  # 高品質化
                "super_sampling": False,  # 超解像処理OFF
                "return_fragment": False,
                "ref_free": False  # v4ではprompt_textを使用するため
            }

            logger.info("🔄 音声生成実行中...")

            # TTS実行
            tts_generator = self.tts_pipeline.run(request_params)
            sr, audio_data = next(tts_generator)

            # 音声保存
            sf.write(str(output_path), audio_data, sr)

            # ファイル情報
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            duration = len(audio_data) / sr

            logger.info(f"✅ ボイスクローン成功: {output_path}")
            logger.info(f"📊 ファイルサイズ: {file_size:.2f}MB, 再生時間: {duration:.2f}秒")
            return True

        except Exception as e:
            logger.error(f"❌ ボイスクローンエラー: {str(e)}")
            logger.error(traceback.format_exc())
            return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="GPT-SoVITS シンプル CLI")
    # 実際の呼び出し方に合わせた引数定義: <ref_audio> <target_text> <output>
    parser.add_argument("ref_audio", help="リファレンス音声ファイルパス")
    parser.add_argument("target_text", help="生成したいテキスト")
    parser.add_argument("output", help="出力ファイルパス")
    parser.add_argument("--speed", type=float, default=1.0, help="音声速度 (0.5-2.0, デフォルト: 1.0)")

    args = parser.parse_args()

    try:
        # CLIの初期化
        cli = SimpleGPTSoVITSCLI()

        # モデル初期化
        if not cli.initialize_models():
            logger.error("❌ モデル初期化に失敗しました")
            return 1

        # ボイスクローン実行 (ref_textは空文字列でv4モードでは問題なし)
        success = cli.generate_voice_clone(
            ref_text="",  # v4モードではprompt_textとしてtarget_textを使用するため空でOK
            ref_audio_path=args.ref_audio,
            target_text=args.target_text,
            output_path=args.output,
            speed_factor=args.speed
        )

        if success:
            print(f"✅ ボイスクローン完了: {args.output}")
            return 0
        else:
            print("❌ ボイスクローンに失敗しました")
            return 1

    except KeyboardInterrupt:
        logger.info("⏹ ユーザーによって中断されました")
        return 0
    except Exception as e:
        logger.error(f"実行エラー: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())