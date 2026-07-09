![Every Tongue — AI Bible study for every language](docs/every-tongue-cover.png)

# Every Tongue

**Scripture passage in → grounded study material out → English back-translation → reviewer handout — powered by Gemma on AMD GPU.**

Every Tongue drafts **reviewable study material from existing human Bible
translations** for low-resource languages. **It does not translate Scripture**
and **requires native-speaker review** before any use.

- 🟢 **Live demo:** https://huggingface.co/spaces/exnav29/every-tongue
- 📦 **Repo:** https://github.com/Exnav29/every-tongue
- 🔬 **AMD/Gemma evidence:** [`evidence/amd-gpu/`](evidence/amd-gpu/)
- 🖥️ **Slide deck:** [`docs/Every-Tongue-deck.pdf`](docs/Every-Tongue-deck.pdf)

Built for the **AMD Developer Hackathon: ACT II** — Track 3 (Unicorn).

---

## ⏱️ 60-Second Judge Demo

1. **Open the live demo:** https://huggingface.co/spaces/exnav29/every-tongue
2. **Target translation:** leave **Swahili — Kiswahili Neno 2015** (the default).
3. **Passage reference:** type **`John 3:16-17`**.
4. Click **📖 Fetch passage** — the passage loads in English *and* Swahili,
   pulled from ScriptureFlow (real expert human translations, side by side).
5. Click **✍️ Draft study material** — a grounded study guide is drafted in
   Swahili and English, with phrases flagged for review.
6. Click **🔄 Back-translate draft to English** — verify the meaning survived
   without needing to read Swahili.
7. Click **📄 Export target-language PDF** (or **Export as Markdown**) — get a
   reviewer handout.
8. **Now switch to Akuapem Twi** and run the *same* passage. Output quality
   drops noticeably — this is the **low-resource stress case**, and it shows
   exactly why the grounding + back-translation + native-speaker-review
   guardrails exist.

---

## 🔴 AMD Compute Usage

This is **real GPU inference on AMD hardware**, configured end-to-end on the
AMD Developer Cloud — **not a hosted-API wrapper**. The full trail (notebook,
logs, `rocm-smi` output, benchmarks, generation samples) is committed and
reproducible in **[`evidence/amd-gpu/`](evidence/amd-gpu/)**.

| | |
| --- | --- |
| **Model** | Google **Gemma 3 12B Instruct** (`google/gemma-3-12b-it`, bfloat16) |
| **Hardware** | **AMD Radeon GPU, gfx1100 (RDNA3), ~48 GB VRAM** (51.5 GB reported), on AMD Developer Cloud |
| **Runtime** | **AMD ROCm 7.2** + **vLLM 0.16.1** (Triton attention backend), PyTorch 2.9.1 |
| **Throughput** | **~34.8 tokens/sec** (end-to-end batched) |
| **VRAM used** | **49.2 GB / 51.5 GB** (25.41 GiB model weights resident) |
| **Load time** | **262 s** (first load, incl. download + torch.compile) |

`rocm-smi` confirmed the model resident on the AMD GPU (VRAM jumped from 0% idle
to ~84% loaded — screenshots in the evidence folder).

**Keywords for the record:** AMD · ROCm · Radeon · gfx1100 · RDNA3 · vLLM ·
Gemma 3 12B — all verifiable in this repo.

---

## 🔀 Backend honesty (please read)

Every Tongue has a three-rung AI backend ladder, and we are explicit about
which one does what:

1. **Self-hosted Gemma 3 12B on the AMD Radeon GPU (ROCm + vLLM)** — the real
   AMD compute story. **Documented, benchmarked, and reproducible** in
   [`evidence/amd-gpu/`](evidence/amd-gpu/). This is where our Gemma-on-AMD
   results come from.
2. **Fireworks AI (AMD-backed) stand-in** — **what the public live demo uses
   right now**, for uptime. The demo's status box labels this honestly. Note:
   Fireworks has no serverless Gemma, so the live demo runs a stand-in model,
   **not live Gemma-on-AMD**.
3. **Mock preview** — if no backend is configured, the app returns clearly
   **labeled placeholder text** so the UI is explorable offline.

> The **live demo runs the Fireworks stand-in, not live Gemma-on-AMD.** Our
> genuine Gemma-on-AMD work is the committed evidence in this repo. We won't
> imply otherwise anywhere.

---

## 🧪 Why low-resource languages need guardrails (the thesis)

We ran the same passages through Gemma 3 12B on the AMD GPU in two languages:

- **Swahili** (higher-resource): fluent, coherent, theologically apt.
- **Akuapem Twi** (genuinely low-resource): repetitive, grammatically weak,
  and — without grounding — prone to drifting Scripture references.

**That gap is the argument for the product.** If frontier AI already handled
every low-resource language perfectly, Every Tongue wouldn't need to exist. So
we build in three guardrails: **(1)** ground every draft in real expert human
translations via ScriptureFlow, **(2)** provide an English back-translation so
a non-speaker can verify meaning, and **(3)** flag phrases and require
**native-speaker review** before use. See the side-by-side outputs in
[`evidence/amd-gpu/`](evidence/amd-gpu/).

---

## 🚫 Not Bible translation

Every Tongue **generates study material** (devotionals, study guides,
discussion questions, summaries) **from Scripture that has already been
translated by human experts.** It does **not** translate the Bible, does not
produce Scripture text, and is **not** a production translation system. Every
output is a **draft that requires native-speaker and ministry-leader review.**

---

## 🌍 Language coverage

The dropdown offers **ScriptureFlow's full corpus — 199 translations spanning
125 languages** — tiered honestly:

- **Verified (4):** Swahili, Akuapem Twi, Asante Twi, Ewe — hand-mapped,
  Latin-script, tested end-to-end (fetch → draft → back-translate → export).
- **Experimental (the rest):** the full corpus, so the breadth is real and
  visible. Generation quality varies (the low-resource languages are the hard
  case this tool exists for); reference lookup may fall back to paste-in; and
  PDF export steers non-Latin / right-to-left scripts to Markdown to avoid
  broken output. The tiering **is** the honesty: real breadth, clearly marked
  rigor.

---

## 💻 Run locally

```bash
git clone https://github.com/Exnav29/every-tongue.git
cd every-tongue
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py                                       # serves on http://127.0.0.1:7860
```

**Backend configuration (environment variables — never commit secrets):**

| Variable | Effect |
| --- | --- |
| `FIREWORKS_API_KEY` | Use the Fireworks (AMD-backed) backend for real generation. |
| `GEMMA_MODEL` *(optional)* | Override the model id requested from the backend. |
| *(none set)* | **Mock preview mode** — the app runs and is explorable, returning clearly **labeled placeholder** output. |

A self-hosted vLLM Gemma endpoint (e.g. the AMD Radeon GPU box used for the
[evidence](evidence/amd-gpu/)) is also supported and tried before Fireworks
when configured.

---

## 📎 Submission assets

| Asset | Location |
| --- | --- |
| **Live demo** | https://huggingface.co/spaces/exnav29/every-tongue |
| **GitHub repo** | https://github.com/Exnav29/every-tongue |
| **Slide deck (PDF)** | [`docs/Every-Tongue-deck.pdf`](docs/Every-Tongue-deck.pdf) |
| **Cover image** | [`docs/every-tongue-cover.png`](docs/every-tongue-cover.png) |
| **AMD/Gemma evidence** | [`evidence/amd-gpu/`](evidence/amd-gpu/) |
| **Video walkthrough** | https://drive.google.com/drive/folders/1OVaaGHCQST0raSkZGuVQCIuxruLyi6sH?usp=sharing |
| **Eval scaffold** | _`eval/` — planned; only real, generated outputs will be included_ |

---

## ⚠️ Known limitations

- **Low-resource quality is uneven.** Akuapem Twi is a deliberate stress case;
  its output is weak and must not be used without native-speaker review.
- **The live demo uses a Fireworks stand-in model, not live Gemma-on-AMD.**
  The Gemma-on-AMD results are the committed evidence.
- **Mock mode is preview only** and is clearly labeled as placeholder text.
- **Not a production translation system** — outputs are review-required drafts.
- **Experimental languages** may not support reference lookup (paste-in
  instead) or PDF export (use Markdown for non-Latin / RTL scripts).

---

## Disclosure

Built solo with Claude Code as an AI pair programmer. Built on **ScriptureFlow**,
a pre-existing open multilingual Scripture API by the author. All Every Tongue
application code was written during the event window.

## License

[MIT](LICENSE)
