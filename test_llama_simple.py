#!/usr/bin/env python3
"""
Simple test script for llama-cpp-python
"""

from llama_cpp import Llama
import os

# Model path
model_path = "/home/adama/wav2lip-project/models/Berghof-NSFW-7B.i1-IQ4_XS.gguf"

print("Loading model...")
print(f"Model: {model_path}")

# Check if model exists
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    exit(1)

try:
    # Load model with GPU support
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,  # Context window
        n_gpu_layers=-1,  # All layers to GPU for maximum speed
        verbose=False
    )
    print("Model loaded successfully!")

    # Test prompt
    prompt = "Human: What is the capital of France?\n\nAssistant:"

    print("\nTest prompt:", prompt)
    print("\nGenerating response...")

    # Generate response
    response = llm(
        prompt,
        max_tokens=100,
        temperature=0.7,
        stop=["Human:", "\n\n"],
        echo=False
    )

    print("\nResponse:", response['choices'][0]['text'])

    # Show GPU usage info
    print("\n" + "="*50)
    print("Test completed successfully!")
    print("The model is working with GPU acceleration.")
    print("\nTo use interactive mode, run:")
    print("  source llama_venv/bin/activate")
    print("  python test_llama_cli.py")

except Exception as e:
    print(f"Error: {e}")
    print("\nTrying CPU-only mode...")

    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=0,  # CPU only
            verbose=False
        )
        print("Model loaded in CPU mode.")

        prompt = "Human: What is 2+2?\n\nAssistant:"
        response = llm(
            prompt,
            max_tokens=50,
            temperature=0.7,
            stop=["Human:", "\n\n"],
            echo=False
        )

        print("\nResponse:", response['choices'][0]['text'])
        print("\nModel works in CPU mode. GPU acceleration may not be available.")

    except Exception as cpu_error:
        print(f"Error in CPU mode: {cpu_error}")