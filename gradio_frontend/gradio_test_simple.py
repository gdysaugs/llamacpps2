#!/usr/bin/env python3
"""
Simple Gradio Test Application
"""

import gradio as gr

def simple_function(text):
    return f"Hello, {text}! Gradio is working!"

with gr.Blocks() as demo:
    gr.Markdown("# Simple Gradio Test")

    with gr.Row():
        text_input = gr.Textbox(label="Enter your name")
        text_output = gr.Textbox(label="Output")

    text_input.change(simple_function, text_input, text_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)