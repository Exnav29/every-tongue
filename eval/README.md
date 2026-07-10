# eval/ — evaluation scaffold

An honest, in-progress evaluation harness for Every Tongue's generated study
material.

- **`passages.json`** — the test-case definitions. `run: "yes"` means a real
  saved output exists in `sample_outputs/`; `run: "pending"` means it has not
  been run.
- **`rubric.md`** — the five-criterion scoring rubric (Scripture fidelity,
  target-language fluency, structure, review-flag quality, back-translation
  faithfulness).
- **`results.md`** — what has actually been observed. Numeric scores are
  **pending** (they require qualified native reviewers) and are not fabricated.
- **`sample_outputs/`** — the real outputs generated on AMD hardware
  (`google/gemma-3-12b-it`, Radeon gfx1100, ROCm 7.2 + vLLM 0.16.1). These are
  the same artifacts documented in [`../evidence/amd-gpu/`](../evidence/amd-gpu/).

Nothing in this folder is invented: unrun cases are labeled pending, and no
score is recorded that wasn't actually assessed.
