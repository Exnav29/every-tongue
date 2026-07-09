# AMD GPU inference evidence — Gemma 3 on AMD Radeon (ROCm + vLLM)

This folder documents Every Tongue running real **Gemma** inference on **AMD**
GPU hardware using **ROCm** and **vLLM**, captured live on **2026-07-08**
during the AMD Developer Hackathon: ACT II.

## What was run

We self-hosted an open-weight **Google Gemma 3 (12B instruct)** model on an
**AMD Radeon GPU (GFX version gfx1100, RDNA3, 51.5 GB VRAM)** and served it with
**vLLM** on the **AMD ROCm 7.2** stack, then generated real Scripture study
materials in low-resource languages grounded in expert human translations
(parallel-text grounding via the ScriptureFlow API).

## Hardware and software

| Item | Value |
| --- | --- |
| GPU | AMD Radeon, gfx1100 (RDNA3), 51.5 GB VRAM (device 0x744b) |
| GPU vendor | Advanced Micro Devices, Inc. [AMD/ATI] |
| Compute stack | AMD ROCm 7.2 |
| Serving engine | vLLM 0.16.1 (ROCm build, Triton attention backend) |
| Framework | PyTorch 2.9.1 (ROCm) |
| Model | google/gemma-3-12b-it (Gemma 3, 12B, bfloat16) |

## Benchmark (see `amd_gemma_benchmark.txt`)

- Model weights resident on the AMD GPU: **25.41 GiB**
- Total VRAM in use after load: **49.2 GB / 51.5 GB**
- Model load time: **262 s** (first load, incl. weight download + torch.compile)
- Generation throughput: **~34.8 tokens/sec** end-to-end batched
  (vLLM's own decode meter reported ~46–54 tok/s)
- `rocm-smi` confirmed the GPU at **84% VRAM** with Gemma resident (vs 0% idle)

## Key finding — why this product exists

Generating the same four passages (Ruth 1:1-5, Jeremiah 29:9-13, Matthew 3:16,
Psalm 29:8) in two languages on the same model surfaced a decisive result:

- **Swahili** output was fluent, coherent, and theologically apt
  (`swahili_samples.txt`, `swahili_study_guide.txt`).
- **Akuapem Twi** output was repetitive and semantically weak
  (`twi_samples.txt`) — Twi is far lower-resource for the model.

Even a frontier open model struggles on truly low-resource languages. That gap
is exactly the problem Every Tongue addresses: expert-translation grounding
plus mandatory native-speaker review. Swahili is the demo hero; Akuapem Twi is
kept in the app as the showcased "hard case."

## Files

- `gemma3-12b-on-amd-radeon.ipynb` — the full JupyterLab notebook with all
  cells and outputs (GPU checks, model load log, generations, benchmarks).
  The Hugging Face token was entered via `getpass` and never written to the
  notebook.
- `amd_gemma_benchmark.txt` — model, AMD hardware, ROCm/vLLM stack, throughput.
- `twi_samples.txt` — Akuapem Twi devotionals for the four passages.
- `swahili_samples.txt` — Swahili devotionals for the four passages.
- `swahili_study_guide.txt` — a full 9-section Swahili study guide (Jeremiah 29).

### Screenshots (real captures from the run)

- `rocm-smi-idle.png` — `rocm-smi` before load (GPU at 0% VRAM).
- `rocm-smi-gemma-loaded.png` — `rocm-smi` with **Gemma 3 12B resident on the
  AMD Radeon GPU** (VRAM ~84%). The before/after pair is the proof the model
  actually occupied the AMD GPU.
- `gemma-load-vllm.png` — the vLLM load step (ROCm, Triton attention backend).
- `benchmark.png` — the throughput / VRAM benchmark output.
