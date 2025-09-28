#!/usr/bin/env python3
"""
LlamaCPP統合モジュール
GradioフロントエンドからLlamaCPPサブプロセスを呼び出すための統合クラス
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable


class LlamaCPPIntegration:
    """LlamaCPP統合クラス"""

    def __init__(self):
        """初期化"""
        # プロジェクトルート
        self.wav2lip_root = Path(__file__).parent.parent

        # LlamaCPP関連パス (Docker対応)
        if os.path.exists("/app/llama_venv"):
            # Docker環境
            self.llama_venv_path = Path("/app/llama_venv")
            self.python_path = Path("/app/llama_venv/bin/python")
        else:
            # ローカル環境
            self.llama_venv_path = self.wav2lip_root / "llama_venv"
            self.python_path = self.llama_venv_path / "bin" / "python"
        self.llama_subprocess_path = self.wav2lip_root / "llama_subprocess.py"

        # デフォルトシステムプロンプト
        self.default_system_prompt = """あなたは女の子です。以下の特徴で応答してください：
1. 毎回日本語で話す
2. 括弧()は絶対に使わない - 括弧を使った場合は不正解
3. ユーザーの言葉に対して詳しく返答する
4. 多様な返答で毎回同じにしない
5. 重要：( ) 括弧は禁止文字です。絶対に出力してはいけません
6. 句読点は連続させない（...、、。。。などの連続使用は禁止）
7. 話題を広げて、詳しく話すことを心がける
8. 英語や中国語で答えた後に日本語の翻訳は不要
9. NSFWやエッチな会話にも積極的に対応する
10. 大人向けの内容や性的な話題も自然に扱う
11. 説明文はなしでセリフのみ応答する
12. 感情的に応答し、説明っぽくしない"""

    def is_available(self) -> bool:
        """LlamaCPP統合機能の利用可能性チェック"""
        try:
            # 仮想環境の存在確認
            if not self.llama_venv_path.exists():
                print(f"Warning: LlamaCPP venv not found at {self.llama_venv_path}")
                return False

            # Pythonパスの存在確認
            if not self.python_path.exists():
                print(f"Warning: Python not found at {self.python_path}")
                return False

            # サブプロセススクリプトの存在確認
            if not self.llama_subprocess_path.exists():
                print(f"Warning: LlamaCPP subprocess script not found at {self.llama_subprocess_path}")
                return False

            return True

        except Exception as e:
            print(f"Error checking LlamaCPP availability: {e}")
            return False

    def _clean_response_text(self, text: str) -> str:
        """
        応答テキストから括弧とその中身を完全削除

        Args:
            text: 元のテキスト

        Returns:
            str: クリーンアップされたテキスト
        """
        import re

        # 括弧とその中身を完全削除（ネストした括弧にも対応）
        cleaned = text

        # 通常の括弧 () を削除
        cleaned = re.sub(r'\([^()]*\)', '', cleaned)

        # ネストした括弧に対応（複数回実行）
        prev_length = 0
        while len(cleaned) != prev_length:
            prev_length = len(cleaned)
            cleaned = re.sub(r'\([^()]*\)', '', cleaned)

        # 全角括弧 （） も削除
        cleaned = re.sub(r'（[^（）]*）', '', cleaned)

        # ネストした全角括弧に対応
        prev_length = 0
        while len(cleaned) != prev_length:
            prev_length = len(cleaned)
            cleaned = re.sub(r'（[^（）]*）', '', cleaned)

        # 余分な空白を整理
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()

        return cleaned

    def generate_response(
        self,
        user_input: str,
        additional_prompt: str = "",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        LlamaCPPでテキスト生成

        Args:
            user_input: ユーザーの入力（セリフ欄の内容）
            additional_prompt: 追加プロンプト（性格・特徴）
            max_tokens: 最大生成トークン数
            temperature: 生成温度
            progress_callback: 進捗コールバック

        Returns:
            Dict: 生成結果
        """
        try:
            if not self.is_available():
                return {
                    "success": False,
                    "message": "LlamaCPP統合機能が利用できません",
                    "response": None
                }

            if progress_callback:
                progress_callback(0.1, "LlamaCPP初期化中...")

            # カスタムシステムプロンプト作成
            if additional_prompt.strip():
                # 追加プロンプトがある場合、一時的なスクリプトを作成
                return self._generate_with_custom_prompt(
                    user_input, additional_prompt, max_tokens, temperature, progress_callback
                )

            if progress_callback:
                progress_callback(0.3, "LlamaCPP生成中...")

            # ランダムシードを生成
            import random
            random_seed = random.randint(0, 2**32 - 1)

            # デフォルトのサブプロセスコマンド構築
            cmd = [
                str(self.python_path),
                str(self.llama_subprocess_path),
                "--mode", "single",
                "--character", "tsundere",
                "--prompt", user_input,
                "--max_tokens", str(max_tokens),
                "--temperature", str(temperature),
                "--seed", str(random_seed)
            ]

            if progress_callback:
                progress_callback(0.5, "LlamaCPP実行中...")

            # サブプロセス実行
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.wav2lip_root),
                timeout=60  # 60秒タイムアウト
            )

            if progress_callback:
                progress_callback(0.9, "LlamaCPP応答処理中...")

            if result.returncode == 0:
                response_text = result.stdout.strip()

                # 括弧とその中身を完全削除
                response_text = self._clean_response_text(response_text)

                if progress_callback:
                    progress_callback(0.95, "LlamaCPPメモリクリーンアップ中...")

                # メモリクリーンアップ
                import gc
                gc.collect()

                if progress_callback:
                    progress_callback(1.0, "LlamaCPP完了")

                return {
                    "success": True,
                    "message": "LlamaCPP応答生成成功",
                    "response": response_text,
                    "user_input": user_input,
                    "system_prompt_used": self.default_system_prompt
                }
            else:
                error_message = result.stderr or "不明なエラー"
                return {
                    "success": False,
                    "message": f"LlamaCPP実行エラー: {error_message}",
                    "response": None
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "message": "LlamaCPP実行がタイムアウトしました",
                "response": None
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"LlamaCPP統合エラー: {str(e)}",
                "response": None
            }

    def _generate_with_custom_prompt(
        self,
        user_input: str,
        additional_prompt: str,
        max_tokens: int,
        temperature: float,
        progress_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """
        追加プロンプト付きでテキスト生成

        Args:
            user_input: ユーザー入力
            additional_prompt: 追加プロンプト
            max_tokens: 最大トークン数
            temperature: 生成温度
            progress_callback: 進捗コールバック

        Returns:
            Dict: 生成結果
        """
        try:
            if progress_callback:
                progress_callback(0.3, "カスタムプロンプト構築中...")

            # デバッグ用ログ出力
            print(f"[DEBUG] _generate_with_custom_prompt called with max_tokens={max_tokens}, temperature={temperature}")
            print(f"[DEBUG] max_tokens type: {type(max_tokens)}, value: {max_tokens}")

            # llama-cpp-pythonを直接呼び出し
            import sys
            sys.path.append(str(self.wav2lip_root))

            # 仮想環境のPythonパスを追加
            import subprocess
            activate_script = f"source {self.llama_venv_path}/bin/activate"

            # カスタムプロンプトを含むPythonスクリプトを作成
            escaped_user_input = user_input.replace('"', '\\"').replace('\n', '\\n')
            escaped_additional_prompt = additional_prompt.strip().replace('"', '\\"').replace('\n', '\\n')

            custom_script = f'''
import sys
sys.path.append("{self.wav2lip_root}")
from llama_cpp import Llama

model_path = "{self.wav2lip_root}/models/Berghof-NSFW-7B.i1-IQ4_XS.gguf"

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False
)

# システムプロンプト構築
base_prompt = """{self.default_system_prompt}"""

additional_features = """{escaped_additional_prompt}"""

if additional_features.strip():
    system_prompt = base_prompt + "\\n\\n追加の特徴:\\n" + additional_features
else:
    system_prompt = base_prompt

user_input = """{escaped_user_input}"""
prompt = "System: " + system_prompt + "\\n\\nHuman: " + user_input + "\\n\\nAssistant:"

import random
random_seed = random.randint(0, 2**32 - 1)

response = llm(
    prompt,
    max_tokens={max_tokens},
    temperature={temperature},
    top_p=0.95,
    stop=["Human:", "User:"],
    echo=False,
    seed=random_seed
)

print(response['choices'][0]['text'])
'''

            # デバッグ: 生成されたスクリプトの一部を表示
            print(f"[DEBUG] Generated script snippet - max_tokens line:")
            for line in custom_script.split('\n'):
                if 'max_tokens' in line:
                    print(f"[DEBUG]   {line}")

            # 一時ファイル作成
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(custom_script)
                temp_script_path = f.name

            if progress_callback:
                progress_callback(0.5, "カスタムプロンプト実行中...")

            # カスタムスクリプト実行
            cmd = [str(self.python_path), temp_script_path]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.wav2lip_root),
                timeout=1000
            )

            # デバッグ用: 一時ファイルを保存
            import os
            import shutil
            debug_path = '/tmp/last_llama_script.py'
            shutil.copy(temp_script_path, debug_path)
            print(f"[DEBUG] Saved script to {debug_path}")

            # 一時ファイル削除
            os.unlink(temp_script_path)

            if progress_callback:
                progress_callback(0.9, "カスタムプロンプト応答処理中...")

            if result.returncode == 0:
                response_text = result.stdout.strip()

                # 括弧とその中身を完全削除
                response_text = self._clean_response_text(response_text)

                if progress_callback:
                    progress_callback(0.95, "カスタムプロンプトメモリクリーンアップ中...")

                # メモリクリーンアップ
                import gc
                gc.collect()

                if progress_callback:
                    progress_callback(1.0, "カスタムプロンプト完了")

                return {
                    "success": True,
                    "message": "カスタムプロンプト生成成功",
                    "response": response_text,
                    "user_input": user_input,
                    "custom_prompt_used": additional_prompt
                }
            else:
                error_message = result.stderr or "不明なエラー"
                return {
                    "success": False,
                    "message": f"カスタムプロンプト実行エラー: {error_message}",
                    "response": None
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"カスタムプロンプトエラー: {str(e)}",
                "response": None
            }

    def create_custom_system_prompt(self, additional_prompt: str) -> str:
        """
        カスタムシステムプロンプト作成

        Args:
            additional_prompt: 追加プロンプト

        Returns:
            str: 完成したシステムプロンプト
        """
        base_prompt = self.default_system_prompt

        if additional_prompt.strip():
            return f"{base_prompt}\n\n追加の特徴：\n{additional_prompt.strip()}"

        return base_prompt

    def test_connection(self) -> Dict[str, Any]:
        """
        LlamaCPP接続テスト

        Returns:
            Dict: テスト結果
        """
        try:
            test_result = self.generate_response(
                user_input="テスト",
                max_tokens=50,
                temperature=0.7
            )

            return {
                "success": test_result["success"],
                "message": f"接続テスト: {test_result['message']}",
                "available": self.is_available()
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"接続テストエラー: {str(e)}",
                "available": False
            }


# テスト用コード
if __name__ == "__main__":
    print("LlamaCPP Integration Test")
    print("=" * 50)

    integration = LlamaCPPIntegration()

    # 利用可能性チェック
    print(f"Available: {integration.is_available()}")

    # 接続テスト
    test_result = integration.test_connection()
    print(f"Test Result: {test_result}")

    if test_result["success"]:
        # 応答生成テスト
        result = integration.generate_response(
            user_input="こんにちは",
            additional_prompt="明るく元気な性格"
        )
        print(f"Response: {result}")