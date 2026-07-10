# Evaluation results

**Honesty note:** this file records only what has actually been observed. Formal
per-criterion numeric scoring by qualified native speakers has **not** been run
yet — those cells are marked *pending*, not filled with invented numbers.

## What has actually been run

Nine cases were generated on real AMD hardware (`google/gemma-3-12b-it`, AMD
Radeon gfx1100, ROCm 7.2 + vLLM 0.16.1) during the documented notebook run —
see [`../evidence/amd-gpu/`](../evidence/amd-gpu/). Raw outputs are in
[`sample_outputs/`](sample_outputs/).

## Qualitative finding (observed, not yet rubric-scored)

The decisive, reproducible observation from the run:

- **Swahili** (higher-resource): fluent, coherent, theologically apt; quoted
  Scripture matched the ScriptureFlow source. Strong across devotionals and the
  Jeremiah 29 study guide.
- **Akuapem Twi** (genuinely low-resource): noticeably weaker — repetitive
  phrasing, degraded grammar/idiom. This is the **showcased hard case** and the
  reason the product's guardrails (grounding + back-translation + mandatory
  native-speaker review) exist.

This Swahili-vs-Twi gap is qualitative and directly visible in the sample files;
it has **not** been converted into per-criterion 1–5 scores because that
requires qualified native reviewers (see `rubric.md`).

## Scorecard

| Case (`passages.json` id) | Run | Fidelity | Fluency | Structure | Flags | Back-trans |
| --- | --- | --- | --- | --- | --- | --- |
| `*-swahili-devotional` (×4) | yes | pending | pending | pending | pending | pending |
| `*-twi-devotional` (×4) | yes | pending | pending | pending | pending | pending |
| `jeremiah-29-9-13-swahili-studyguide` | yes | pending | pending | pending | pending | pending |
| all `pending` cases in `passages.json` | no | — | — | — | — | — |

**Next step:** recruit native Swahili and Akuapem Twi reviewers to score the
nine run cases against `rubric.md`, then extend to the pending cases.
