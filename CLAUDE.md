# Every Tongue — repo guide

Every Tongue drafts reviewable Scripture **study material** (devotionals, study
guides, discussion questions, summaries) from **existing human Bible
translations** for low-resource languages. It does **not** translate Scripture
and **requires native-speaker review**. It is built on **ScriptureFlow** (a
pre-existing open multilingual Scripture API by the author) for the AMD
Developer Hackathon: ACT II — Track 3 (Unicorn).

## Non-negotiable accuracy constraints — honesty is this project's identity

- The AMD hardware used was an **AMD Radeon GPU, gfx1100 (RDNA3), ~48 GB VRAM**.
  Use only these real identifiers — do **not** describe it as a larger
  data-center GPU class or overstate the VRAM.
- Real benchmark: `google/gemma-3-12b-it`, **ROCm 7.2, vLLM 0.16.1, ~34.8
  tokens/sec, 49.2/51.5 GB VRAM, 262 s load.** Never invent or inflate numbers.
- The **live deployed demo runs a Fireworks (AMD-backed) stand-in model,
  honestly labeled — NOT live Gemma-on-AMD.** The genuine Gemma-on-AMD work is
  the documented evidence in `evidence/amd-gpu/`. Never let the README, UI, or
  status box imply the live demo runs Gemma-on-AMD.
- Every Tongue **drafts study material from existing translations; it does not
  translate Scripture.** Keep that framing everywhere.

## Working style

- **`app.py` is the LIVE, submitted demo.** After ANY change to it, smoke-test
  the full path (Swahili → fetch → draft → back-translate → export PDF/Markdown)
  before continuing. Prefer adding visible copy over rewiring interactive
  elements. **Do not refactor `app.py` into modules.** Commit `app.py` changes
  **separately** from documentation so a UI problem can be reverted without
  losing doc work.
- The maintainer is a coding beginner — explain in plain language, work step by
  step, and pause for confirmation at each milestone (something runs, something
  is pushed).
