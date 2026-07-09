# Every Tongue — slide deck outline (honest, judge-ready)

This is the corrected content outline for the pitch deck. The committed PDF
(`Every-Tongue-deck.pdf`) is the current export; **regenerate it from this
outline** to apply the two corrections flagged at the bottom.

**Numbers to use everywhere (real, do not inflate):**
- Corpus: **199 translations across 125 languages** (ScriptureFlow). *(Not
  "50+ languages" and not "199 languages" — 125 distinct languages, 199
  translations.)*
- Hardware: **AMD Radeon GPU, gfx1100 (RDNA3), ~48 GB VRAM** on AMD Developer
  Cloud. Use only these real identifiers — do not overstate the GPU class or
  VRAM.
- Runtime: **ROCm 7.2 + vLLM 0.16.1**, PyTorch 2.9.1, Triton attention backend.
- Model: **Google Gemma 3 12B Instruct** (`google/gemma-3-12b-it`, bf16).
- Benchmark: **~34.8 tokens/sec**, **49.2 / 51.5 GB VRAM** used, **262 s** load.

---

1. **Title / hook** — Every Tongue: "Scripture passage in → grounded study
   material out → English back-translation → reviewer handout — powered by
   Gemma on AMD GPU." Track 3 (Unicorn). Live & deployed.
2. **Problem** — 7,000+ languages; mainstream AI performs well in fewer than
   ~100. Ministry workers in low-resource languages hand-make every devotional
   or go without.
3. **Why generic AI is risky here** — LLMs trained on internet-scale text are
   fluent in major languages and unreliable in low-resource ones, including
   drifting/fabricated Scripture references — unacceptable for ministry use.
4. **The solution: grounded drafting (not translation)** — pull expert human
   translations (English + target) from ScriptureFlow, draft study material
   grounded in them, flag phrases for review. **It does not translate
   Scripture.**
5. **Demo flow** — select language → enter reference → fetch parallel text →
   generate bilingual study material → English back-translation → export
   reviewer handout. Live URL judges can use.
6. **Architecture** — Gradio app (HF Spaces) · ScriptureFlow data layer · AI
   backend ladder: self-hosted Gemma-on-AMD (evidence) → Fireworks stand-in
   (live demo) → labeled mock. **Be explicit the live demo uses the stand-in.**
7. **AMD / Gemma evidence** — real gfx1100 numbers (above); notebook, logs,
   rocm-smi screenshots, samples in `evidence/amd-gpu/`. Real GPU inference on
   AMD, reproducible — not a hosted-API wrapper.
8. **ScriptureFlow** — the multilingual Scripture data platform beneath the
   product: 199 translations across 125 languages; expert human translations
   as high-quality parallel text for grounding.
9. **Swahili vs. Twi finding (the thesis)** — Gemma handles Swahili well,
   Akuapem Twi poorly; the gap is the argument for grounding + back-translation
   + native-speaker review.
10. **Guardrails & market** — three guardrails; faith-sector beachhead →
    NGO/government localization; free-for-individuals + org subscription +
    API licensing.
11. **Next steps / ask** — expand verified languages, broaden ScriptureFlow,
    optional live Gemma-on-AMD endpoint during judging.

---

## ⚠️ Remaining correction for the deck

- ✅ **Language numbers** — fixed in the current deck ("199 translations across
  125 languages"). A few slides still say "50+ languages" in passing; harmless
  (125 *is* 50+), but tighten to "125 languages" for consistency if re-exporting.
- ⚠️ **Live-demo honesty (still present).** Slide 11 says the running demo lets
  judges "observe grounded multilingual generation on AMD ROCm hardware," which
  implies the **live demo** runs on AMD. It doesn't — the live demo uses the
  **Fireworks stand-in**; the real Gemma-on-AMD run is the committed evidence.
  Reword so the AMD-inference claim points at the evidence and the live demo is
  described as the stand-in (matches the README's backend-honesty note).
