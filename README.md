# Every Tongue

An AI assistant that helps Bible translators and ministry workers draft and
check Scripture **study materials** (devotionals, discussion questions,
summaries — not Bible text itself) in low-resource languages.

Built for the **AMD Developer Hackathon: ACT II** on lablab.ai (Unicorn Track),
July 6–11, 2026.

## Features (planned)

1. **Draft assist** — pick a target language (Swahili, Haitian Creole, or
   Tagalog), paste a passage or study question, and Gemma drafts the study
   material in that language, flagging phrases that need native-speaker review.
2. **Back-translation check** — one button translates the draft back to
   English so a non-speaker can verify the meaning survived.

## Tech

- Python + Gradio web interface
- Gemma (open-weight model), self-hosted with vLLM on an AMD Instinct MI300X
  in AMD Developer Cloud, with Fireworks AI as fallback
- Docker container, deployed to Hugging Face Spaces

## Setup

_Coming soon — instructions will be added as the app is built during the
hackathon week._

## License

[MIT](LICENSE)
