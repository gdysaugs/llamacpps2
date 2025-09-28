#!/usr/bin/env python3
"""
Interactive CLI test for llama-cpp-python with deep thinking capabilities
"""

from llama_cpp import Llama
import sys
import os

def main():
    # Model path
    model_path = "/home/adama/wav2lip-project/models/Berghof-NSFW-7B.i1-IQ4_XS.gguf"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("Loading model... This may take a moment.")

    # Initialize Llama with GPU support
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,  # Context window
            n_gpu_layers=-1,  # All layers to GPU for maximum speed
            n_threads=4,  # CPU threads
            verbose=False,
            seed=42,
            f16_kv=True,  # Use half precision for key/value cache
            use_mlock=False,  # Don't lock memory (can cause issues on some systems)
        )
        print("Model loaded successfully!")
        print(f"Model: {model_path}")
        print("Type 'exit' or 'quit' to end the conversation")
        print("Type 'clear' to clear the conversation history")
        print("-" * 50)

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Conversation history
    messages = []

    # System prompt for character
    system_prompt = """あなたは女の子です。以下の特徴で応答してください：
1. 日本語で自然に話す
2. 括弧()は絶対に使わない
3. 心の声や説明は一切書かず、セリフだけで表現する
4. ユーザーの言葉に対して返答する
5. 多様な返答で毎回同じにしない"""

    messages.append({"role": "system", "content": system_prompt})

    while True:
        try:
            # Get user input
            try:
                user_input = input("\nYou: ").strip()
            except EOFError:
                print("\nUse 'python test_llama_cli.py' directly in terminal for interactive mode.")
                break

            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break

            if user_input.lower() == 'clear':
                messages = [{"role": "system", "content": system_prompt}]
                print("Conversation cleared.")
                continue

            if not user_input:
                continue

            # Add user message to history
            messages.append({"role": "user", "content": user_input})

            # Create prompt from conversation history
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    prompt += f"Human: {msg['content']}\n\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n\n"
            prompt += "Assistant: "

            # Generate response
            print("\nAssistant: ", end="", flush=True)

            response = llm(
                prompt,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.1,
                stop=["Human:", "User:", "\n\n"],
                stream=True,
                echo=False,
            )

            # Stream the response
            full_response = ""
            for chunk in response:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    token = chunk['choices'][0].get('text', '')
                    print(token, end="", flush=True)
                    full_response += token

            print()  # New line after response

            # Add assistant response to history
            messages.append({"role": "assistant", "content": full_response.strip()})

            # Keep conversation history manageable (last 10 exchanges)
            if len(messages) > 21:  # 1 system + 10 exchanges (user + assistant)
                messages = [messages[0]] + messages[-20:]

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit.")
            continue
        except Exception as e:
            print(f"\nError: {e}")
            continue

if __name__ == "__main__":
    main()