"""Every Tongue — hello-world Gradio app.

Day 0 sanity check: proves Python + Gradio + browser all work together.
The real features arrive during the hackathon week (July 6-11, 2026).
"""

import gradio as gr

LANGUAGES = ["Akuapem Twi", "Asante Twi", "Ewe", "Swahili"]


def hello(name: str, language: str) -> str:
    who = name.strip() or "friend"
    return (
        f"Hello, {who}! Every Tongue is up and running.\n\n"
        f"Soon this app will draft Scripture study materials in {language}."
    )


with gr.Blocks(title="Every Tongue") as demo:
    gr.Markdown("# Every Tongue\n*AI-drafted Scripture study materials in low-resource languages*")
    name_box = gr.Textbox(label="Your name")
    language_dropdown = gr.Dropdown(LANGUAGES, value="Akuapem Twi", label="Target language")
    greet_button = gr.Button("Say hello")
    output_box = gr.Textbox(label="Output", lines=3)
    greet_button.click(hello, inputs=[name_box, language_dropdown], outputs=output_box)

if __name__ == "__main__":
    demo.launch()
