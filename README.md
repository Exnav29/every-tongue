# Every Tongue

An AI assistant that generates Scripture **study materials** (study guides,
devotionals, discussion questions — not Bible text itself) in low-resource
languages, for pastors, ministry workers, and translators. Every Tongue is the
first product built on **ScriptureFlow**, an open multilingual Scripture API.

Built for the **AMD Developer Hackathon: ACT II** on lablab.ai (Unicorn Track),
July 6–11, 2026.

## Features

1. **Passage mode** — pick a target language and an English source translation,
   enter a reference (e.g. John 3:16), and the app fetches the passage in both
   languages from ScriptureFlow. Gemma then drafts study material grounded in
   the expert human translation (parallel-text grounding), in the target
   language *and* English side by side, flagging phrases for native-speaker
   review.
2. **Material types** — Study guide, Devotional, Discussion questions, and
   Quick Read, each with its own structure; an audience selector (Children →
   Seniors) shapes tone and reading level independently.
3. **Back-translation check** — translate the draft back to English so a
   non-speaker can verify the meaning survived.
4. **Polished export** — download the study guide as a formatted PDF (or
   Markdown), in the ScriptureFlow study-guide style.

**Demo hero language: Swahili.** Akuapem Twi is kept as a showcased "hard case"
— see the AMD evidence below for why.

## Tech

- Python + Gradio web interface
- **ScriptureFlow** API as the multilingual Scripture data layer
- **Gemma** (open-weight model) as the AI layer. Proven running on **AMD GPU
  hardware via ROCm + vLLM** (see [AMD evidence](evidence/amd-gpu/README.md)),
  with the Fireworks AI API as a fallback.
- Deployed to Hugging Face Spaces

## AMD + Gemma

Every Tongue runs real **Gemma 3 (12B)** inference on an **AMD Radeon GPU
(gfx1100, RDNA3)** using the **AMD ROCm 7.2** stack and **vLLM**. Full logs,
benchmarks (throughput, VRAM, `rocm-smi`), the JupyterLab notebook, and sample
generations are in **[`evidence/amd-gpu/`](evidence/amd-gpu/README.md)**.

The same runs surfaced our core thesis: even a frontier open model is fluent in
higher-resource Swahili but weak in low-resource Akuapem Twi — which is exactly
why expert-translation grounding and mandatory native-speaker review matter.

## Setup

_Full setup and usage instructions are added at feature freeze (Thursday)._

## Disclosure

Built solo with Claude Code as an AI pair programmer. Built on ScriptureFlow, a
pre-existing open multilingual Scripture API by the author. All Every Tongue
application code was written during the event window.

## License

[MIT](LICENSE)
