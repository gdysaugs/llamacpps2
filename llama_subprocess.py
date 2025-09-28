#!/usr/bin/env python3
"""
Subprocess-compatible Llama wrapper
サブプロセスで呼び出し可能なLlamaインターフェース
"""

import sys
import json
import argparse
from llama_cpp import Llama
import os

class LlamaSubprocess:
    def __init__(self, model_path=None, character_mode="default"):
        self.model_path = model_path or "/home/adama/wav2lip-project/models/Berghof-NSFW-7B.i1-IQ4_XS.gguf"
        self.character_mode = character_mode
        self.llm = None

        # キャラクター設定
        self.system_prompts = {
            "default": "あなたは親切で知識豊富なAIアシスタントです。",
            "tsundere": """あなたは女の子です。以下の特徴で応答してください：
1. 日本語で自然に話す
2. 括弧()は絶対に使わない
3. ユーザーの言葉に対して詳しく返答する
4. 多様な返答で毎回同じにしない
5. 話題を広げて、詳しく話すことを心がける""",
            "friendly": "あなたは明るく元気な友達のような話し方をするAIです。親しみやすく、フレンドリーに応答してください。"
        }

    def load_model(self):
        """モデルをロード"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=4096,
            n_gpu_layers=-1,  # 全レイヤーGPU使用
            n_threads=4,
            verbose=False,
            f16_kv=True,
            use_mlock=False,
        )

    def generate(self, prompt, max_tokens=512, temperature=0.7, stream=False, seed=-1):
        """テキスト生成"""
        if self.llm is None:
            self.load_model()

        # システムプロンプトを追加
        system_prompt = self.system_prompts.get(self.character_mode, self.system_prompts["default"])
        full_prompt = f"System: {system_prompt}\n\nHuman: {prompt}\n\nAssistant:"

        if stream:
            return self.llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.1,
                stop=["Human:", "User:", "\n\n"],
                stream=True,
                echo=False,
                seed=seed if seed != -1 else None
            )
        else:
            response = self.llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.1,
                stop=["Human:", "User:", "\n\n"],
                echo=False,
                seed=seed if seed != -1 else None
            )
            return response['choices'][0]['text']

def main():
    parser = argparse.ArgumentParser(description='Llama Subprocess Interface')
    parser.add_argument('--mode', choices=['single', 'interactive', 'json'],
                       default='single', help='実行モード')
    parser.add_argument('--prompt', type=str, help='入力プロンプト')
    parser.add_argument('--max_tokens', type=int, default=200, help='最大トークン数')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')
    parser.add_argument('--seed', type=int, default=-1, help='ランダムシード (-1でランダム)')
    parser.add_argument('--model_path', type=str, help='モデルパス')
    parser.add_argument('--character', choices=['default', 'tsundere', 'friendly'],
                       default='default', help='キャラクター設定')

    args = parser.parse_args()

    try:
        llama_proc = LlamaSubprocess(args.model_path, args.character)

        if args.mode == 'single':
            # 単発生成モード
            if not args.prompt:
                print("Error: --prompt is required for single mode", file=sys.stderr)
                sys.exit(1)

            # ランダムシードを生成（-1の場合）
            import random
            seed = args.seed if args.seed != -1 else random.randint(0, 2**32 - 1)

            response = llama_proc.generate(
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=seed
            )
            print(response)

        elif args.mode == 'interactive':
            # インタラクティブモード（標準入力から読み取り）
            llama_proc.load_model()
            print("Ready for input. Type 'EXIT' to quit.", file=sys.stderr)

            while True:
                try:
                    line = input()
                    if line.strip() == 'EXIT':
                        break

                    response = llama_proc.generate(
                        line,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature
                    )
                    print(f"RESPONSE: {response}")
                    print("END_RESPONSE")
                    sys.stdout.flush()

                except EOFError:
                    break
                except KeyboardInterrupt:
                    break

        elif args.mode == 'json':
            # JSON入出力モード
            llama_proc.load_model()

            while True:
                try:
                    line = input()
                    if line.strip() == 'EXIT':
                        break

                    request = json.loads(line)
                    prompt = request.get('prompt', '')
                    max_tokens = request.get('max_tokens', args.max_tokens)
                    temperature = request.get('temperature', args.temperature)

                    response = llama_proc.generate(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )

                    result = {
                        'response': response,
                        'status': 'success'
                    }
                    print(json.dumps(result, ensure_ascii=False))
                    sys.stdout.flush()

                except EOFError:
                    break
                except KeyboardInterrupt:
                    break
                except json.JSONDecodeError as e:
                    error_result = {
                        'error': f'JSON decode error: {str(e)}',
                        'status': 'error'
                    }
                    print(json.dumps(error_result, ensure_ascii=False))
                    sys.stdout.flush()

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()