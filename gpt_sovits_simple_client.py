#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS シンプルクライアント
api_v2.pyサーバーを使用した音声生成クライアント

使用方法:
1. api_v2サーバー起動 (別ターミナル):
   cd gpt_sovits_full && source ../gpt_sovits_env/bin/activate && python api_v2.py -c GPT_SoVITS/configs/tts_infer.yaml

2. クライアント実行:
   python gpt_sovits_simple_client.py "生成したいテキスト" output.wav
"""

import requests
import argparse
import time
import logging
from pathlib import Path
from typing import Optional
import json

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPTSoVITSClient:
    """GPT-SoVITS API クライアント"""

    def __init__(self, api_url: str = "http://127.0.0.1:9880"):
        """
        初期化

        Args:
            api_url (str): API サーバーのURL
        """
        self.api_url = api_url
        self.project_root = Path(__file__).parent.absolute()

    def wait_for_server(self, timeout: int = 30) -> bool:
        """
        サーバーの起動を待機

        Args:
            timeout (int): タイムアウト秒数

        Returns:
            bool: サーバーが利用可能かどうか
        """
        logger.info(f"🔄 API サーバーの起動を待機中: {self.api_url}")

        for i in range(timeout):
            try:
                response = requests.get(f"{self.api_url}/docs", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ API サーバーが利用可能です")
                    return True
            except requests.exceptions.RequestException:
                pass

            if i < timeout - 1:
                time.sleep(1)
                logger.info(f"⏳ 待機中... ({i+1}/{timeout})")

        logger.error("❌ API サーバーが利用できません")
        return False

    def generate_audio(self,
                      text: str,
                      output_path: str = "output/generated.wav",
                      ref_audio_path: str = None,
                      prompt_text: str = None,
                      text_lang: str = "ja",
                      prompt_lang: str = "ja",
                      **kwargs) -> bool:
        """
        音声生成

        Args:
            text (str): 生成するテキスト
            output_path (str): 出力ファイルパス
            ref_audio_path (str): リファレンス音声パス
            prompt_text (str): リファレンス音声のテキスト
            text_lang (str): テキストの言語
            prompt_lang (str): プロンプトの言語
            **kwargs: その他のパラメータ

        Returns:
            bool: 成功したかどうか
        """
        try:
            logger.info(f"🎤 音声生成開始: '{text[:50]}{'...' if len(text) > 50 else ''}'")

            # デフォルトリファレンス音声の設定
            if ref_audio_path is None:
                # 利用可能なリファレンス音声を探す
                models_root = self.project_root / "models/gpt_sovits"
                ref_candidates = [
                    "hajimemashite_reference.wav",
                    "hajimemashite_reference.mp3",
                    "baka_reference.wav",
                    "test_reference.wav",
                    "reference.wav"
                ]

                for candidate in ref_candidates:
                    candidate_path = models_root / candidate
                    if candidate_path.exists():
                        ref_audio_path = str(candidate_path)
                        break

                if ref_audio_path is None:
                    logger.error("❌ 利用可能なリファレンス音声が見つかりません")
                    return False

            if prompt_text is None:
                prompt_text = "はじめまして、こんにちは"

            logger.info(f"📻 リファレンス音声: {ref_audio_path}")
            logger.info(f"💭 プロンプトテキスト: {prompt_text}")

            # 出力ディレクトリの作成
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # APIリクエストパラメータ
            request_data = {
                "text": text,
                "text_lang": text_lang,
                "ref_audio_path": ref_audio_path,
                "prompt_text": prompt_text,
                "prompt_lang": prompt_lang,
                "top_k": kwargs.get("top_k", 15),
                "top_p": kwargs.get("top_p", 1.0),
                "temperature": kwargs.get("temperature", 1.0),
                "text_split_method": kwargs.get("text_split_method", "cut5"),
                "batch_size": kwargs.get("batch_size", 1),
                "speed_factor": kwargs.get("speed_factor", 1.0),
                "seed": kwargs.get("seed", -1),
                "parallel_infer": kwargs.get("parallel_infer", True),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.35),
                "streaming_mode": False,
                "media_type": "wav",
                "ref_free": True
            }

            logger.info(f"📝 リクエストパラメータ:")
            for key, value in request_data.items():
                if key != "text":
                    logger.info(f"  {key}: {value}")

            # API呼び出し
            logger.info("🔄 API 呼び出し中...")
            response = requests.post(
                f"{self.api_url}/tts",
                json=request_data,
                timeout=120
            )

            if response.status_code == 200:
                # 音声データの保存
                with open(output_path, 'wb') as f:
                    f.write(response.content)

                # ファイル情報の表示
                file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"✅ 音声生成成功: {output_path}")
                logger.info(f"📊 ファイルサイズ: {file_size:.2f}MB")
                return True
            else:
                try:
                    error_info = response.json()
                    logger.error(f"❌ API エラー ({response.status_code}): {error_info}")
                except:
                    logger.error(f"❌ API エラー ({response.status_code}): {response.text[:200]}")
                return False

        except Exception as e:
            logger.error(f"❌ 音声生成エラー: {str(e)}")
            return False

    def check_server_status(self) -> dict:
        """サーバーステータスの確認"""
        try:
            response = requests.get(f"{self.api_url}/docs", timeout=5)
            return {"status": "online", "code": response.status_code}
        except Exception as e:
            return {"status": "offline", "error": str(e)}


def start_api_server_background():
    """バックグラウンドでAPI サーバーを起動"""
    import subprocess
    import os

    logger.info("🚀 API サーバーをバックグラウンドで起動中...")

    # gpt_sovits_fullディレクトリに移動してapi_v2.pyを実行
    project_root = Path(__file__).parent.absolute()
    gpt_sovits_full_dir = project_root / "gpt_sovits_full"
    env_activate = project_root / "gpt_sovits_env/bin/activate"

    # 起動コマンドの作成
    cmd = f"cd {gpt_sovits_full_dir} && source {env_activate} && python api_v2.py -c GPT_SoVITS/configs/tts_infer.yaml"

    # バックグラウンドで実行
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid if hasattr(os, 'setsid') else None
    )

    logger.info(f"📡 API サーバープロセス ID: {process.pid}")
    return process


def main():
    """コマンドライン実行用メイン関数"""
    parser = argparse.ArgumentParser(description="GPT-SoVITS クライアント")
    parser.add_argument("text", help="生成するテキスト")
    parser.add_argument("output", help="出力ファイル名")
    parser.add_argument("--ref_audio", help="リファレンス音声パス")
    parser.add_argument("--prompt_text", help="リファレンス音声のテキスト")
    parser.add_argument("--text_lang", default="ja", help="テキストの言語 (デフォルト: ja)")
    parser.add_argument("--prompt_lang", default="ja", help="プロンプトの言語 (デフォルト: ja)")
    parser.add_argument("--api_url", default="http://127.0.0.1:9880", help="API URL")
    parser.add_argument("--auto_start", action="store_true", help="自動的にサーバーを起動")

    args = parser.parse_args()

    try:
        # クライアントの初期化
        client = GPTSoVITSClient(api_url=args.api_url)

        # サーバーの自動起動
        server_process = None
        if args.auto_start:
            server_process = start_api_server_background()
            time.sleep(5)  # サーバー起動待機

        # サーバーの起動を待機
        if not client.wait_for_server():
            print("❌ API サーバーが起動していません")
            print("🔧 手動でサーバーを起動してください:")
            print("   cd gpt_sovits_full && source ../gpt_sovits_env/bin/activate && python api_v2.py -c GPT_SoVITS/configs/tts_infer.yaml")
            return 1

        # 音声生成
        success = client.generate_audio(
            text=args.text,
            output_path=args.output,
            ref_audio_path=args.ref_audio,
            prompt_text=args.prompt_text,
            text_lang=args.text_lang,
            prompt_lang=args.prompt_lang
        )

        if success:
            print(f"✅ 音声生成完了: {args.output}")
            return 0
        else:
            print("❌ 音声生成に失敗しました")
            return 1

    except KeyboardInterrupt:
        logger.info("⏹ ユーザーによって中断されました")
        return 0
    except Exception as e:
        logger.error(f"実行エラー: {str(e)}")
        return 1
    finally:
        # サーバープロセスのクリーンアップ
        if 'server_process' in locals() and server_process:
            try:
                server_process.terminate()
                logger.info("🔌 バックグラウンドサーバーを終了しました")
            except:
                pass


# Gradio/Wav2Lip統合用関数
def generate_audio_for_wav2lip(text: str,
                              output_filename: str = "generated_audio.wav",
                              api_url: str = "http://127.0.0.1:9880",
                              **kwargs) -> Optional[str]:
    """
    Wav2Lip統合用の音声生成関数

    Args:
        text (str): 生成するテキスト
        output_filename (str): 出力ファイル名
        api_url (str): API URL
        **kwargs: その他のパラメータ

    Returns:
        Optional[str]: 生成された音声ファイルのフルパス（失敗時はNone）
    """
    try:
        client = GPTSoVITSClient(api_url=api_url)
        project_root = Path(__file__).parent.absolute()
        output_path = project_root / "output" / output_filename

        success = client.generate_audio(
            text=text,
            output_path=str(output_path),
            **kwargs
        )

        return str(output_path) if success else None
    except Exception as e:
        logger.error(f"Wav2Lip統合用音声生成エラー: {str(e)}")
        return None


if __name__ == "__main__":
    exit(main())