"""Every Tongue — AI-drafted Scripture study materials in low-resource languages.

Data layer: ScriptureFlow (pre-existing open API) — parallel-text grounding:
the passage is fetched in BOTH English and the target language, and the pair
is sent to Gemma so drafts match the vocabulary of the expert translation.

AI layer (failure ladder): self-hosted Gemma on AMD MI300X via vLLM when the
endpoint is up -> Fireworks AI -> mock responses (no key configured yet).

Built for the AMD Developer Hackathon: ACT II, July 2026.
"""

import os
import re
import tempfile
from datetime import datetime, timezone

import gradio as gr
import requests

SCRIPTUREFLOW_BASE = "https://scriptureflow-api-preview.pages.dev"
ENGLISH_VERSION = "en-kjv"  # public domain, no inline footnotes
TIMEOUT = 20

FIREWORKS_BASE = "https://api.fireworks.ai/inference"


# AI backends — configured via environment variables, never committed.
# Read at call time so a key pasted mid-session only needs an app restart,
# not a code change.
def fireworks_key() -> str:
    return os.environ.get("FIREWORKS_API_KEY", "")


def mi300x_url() -> str:
    return os.environ.get("MI300X_BASE_URL", "").rstrip("/")

LANGUAGES = {
    "Akuapem Twi": "tw-wakna",
    "Asante Twi": "tw-wasna",
    "Ewe": "ee-oal",
    "Swahili": "swh-onen",
}

MATERIAL_TYPES = ["Devotional", "Discussion questions", "Summary"]

AUDIENCES = ["General Congregation", "Children", "Teens", "Young Adults",
             "Adults", "Seniors"]

# ---------------------------------------------------------------------------
# Book-name mapping.
#
# ScriptureFlow stores each translation's books under LOCAL names (John is
# "Yohane" in Twi, "Yohana" in Swahili, "Yohanes" in Ewe), and the API's own
# English-name mapping is incomplete. These tables were built from each
# version's live /books.json on July 6, 2026. Keys are normalized English
# book ids (lowercase, no spaces/hyphens); values are the local book slugs.
# At runtime we still fetch /books.json and prefer the API's mapping when it
# exists, so API improvements win automatically.
# ---------------------------------------------------------------------------

BOOK_OVERRIDES = {
    "tw-wakna": {
        "1kings": "1ahemfo", "2kings": "2ahemfo",
        "1chronicles": "1beresosɛm", "2chronicles": "2beresosɛm",
        "1corinthians": "1korintofo", "2corinthians": "2korintofo",
        "1peter": "1petro", "2peter": "2petro",
        "1thessalonians": "1tesalonikafo", "2thessalonians": "2tesalonikafo",
        "1john": "1yohane", "2john": "2yohane", "3john": "3yohane",
        "revelation": "adiyisɛm", "acts": "asomafo",
        "judges": "atemmufo", "ephesians": "efesofo",
        "philippians": "filipifo", "galatians": "galatifo",
        "haggai": "hagai", "hebrews": "hebrifo", "colossians": "kolosefo",
        "lamentations": "kwadwom", "luke": "luka", "malachi": "malaki",
        "mark": "marko", "micah": "mika", "proverbs": "mmebusɛm",
        "psalms": "nnwom", "songofsolomon": "nnwommudwom",
        "obadiah": "obadia", "romans": "romafo", "zechariah": "sakaria",
        "zephaniah": "sefania", "james": "yakobo", "jeremiah": "yeremia",
        "isaiah": "yesaia", "john": "yohane", "jonah": "yona",
        "joshua": "yosua", "joel": "yoɛl", "jude": "yuda",
        "ecclesiastes": "ɔsɛnkafo", "ezra": "ɛsra",
        "esther": "ɛster",
    },
    "tw-wasna": {
        "1kings": "1ahemfo", "2kings": "2ahemfo",
        "1chronicles": "1berɛsosɛm", "2chronicles": "2berɛsosɛm",
        "1corinthians": "1korintofoɔ", "2corinthians": "2korintofoɔ",
        "1peter": "1petro", "2peter": "2petro",
        "1thessalonians": "1tesalonikafoɔ", "2thessalonians": "2tesalonikafoɔ",
        "1john": "1yohane", "2john": "2yohane", "3john": "3yohane",
        "revelation": "adiyisɛm", "acts": "asomafoɔ",
        "judges": "atemmufoɔ", "ephesians": "efesofoɔ",
        "philippians": "filipifoɔ", "galatians": "galatifoɔ",
        "haggai": "hagai", "hebrews": "hebrifoɔ", "colossians": "kolosefoɔ",
        "lamentations": "kwadwom", "luke": "luka", "malachi": "malaki",
        "mark": "marko", "micah": "mika", "proverbs": "mmɛbusɛm",
        "psalms": "nnwom", "songofsolomon": "nnwommudwom",
        "obadiah": "obadia", "romans": "romafoɔ", "zechariah": "sakaria",
        "zephaniah": "sefania", "james": "yakobo", "jeremiah": "yeremia",
        "isaiah": "yesaia", "john": "yohane", "jonah": "yona",
        "joshua": "yosua", "joel": "yoɛl", "jude": "yuda",
        "ecclesiastes": "ɔsɛnkafoɔ", "ezra": "ɛsra",
        "esther": "ɛster",
    },
    "ee-oal": {
        "acts": "dɔwɔwɔwo", "ephesians": "efesotɔwo",
        "1kings": "fiawo1", "2kings": "fiawo2",
        "philippians": "filipitɔwo", "galatians": "galatiatɔwo",
        "haggai": "hagai", "songofsolomon": "hawo",
        "hebrews": "hebritɔwo", "ezekiel": "hezekiel",
        "colossians": "kolosetɔwo", "lamentations": "konyifahawo",
        "1corinthians": "korintotɔwo1", "2corinthians": "korintotɔwo2",
        "1chronicles": "kronika1", "2chronicles": "kronika2",
        "proverbs": "lododowo", "luke": "luka", "malachi": "malaki",
        "mark": "marko", "micah": "mika",
        "genesis": "mose1", "exodus": "mose2", "leviticus": "mose3",
        "numbers": "mose4", "deuteronomy": "mose5",
        "ecclesiastes": "nyagblɔla", "revelation": "nyaɖeɖefia",
        "obadiah": "obadia", "1peter": "petro1", "2peter": "petro2",
        "psalms": "psalmowo", "romans": "romatɔwo",
        "1samuel": "samuel1", "2samuel": "samuel2",
        "1thessalonians": "tesalonikatɔwo1", "2thessalonians": "tesalonikatɔwo2",
        "1timothy": "timoteo1", "2timothy": "timoteo2",
        "james": "yakobo", "jeremiah": "yeremia", "isaiah": "yesaya",
        "joel": "yoel", "john": "yohanes",
        "1john": "yohanes1", "2john": "yohanes2", "3john": "yohanes3",
        "jonah": "yona", "joshua": "yosua", "jude": "yuda",
        "zephaniah": "zefania", "zechariah": "zekaria",
        "judges": "ʋɔnudrɔ̃lawo",
    },
    "swh-onen": {
        "1chronicles": "1nyakati", "2chronicles": "2nyakati",
        "1peter": "1petro", "2peter": "2petro",
        "1samuel": "1samweli", "2samuel": "2samweli",
        "1timothy": "1timotheo", "2timothy": "2timotheo",
        "1kings": "1wafalme", "2kings": "2wafalme",
        "1corinthians": "1wakorintho", "2corinthians": "2wakorintho",
        "1thessalonians": "1wathesalonike", "2thessalonians": "2wathesalonike",
        "1john": "1yohana", "2john": "2yohana", "3john": "3yohana",
        "amos": "amosi", "job": "ayubu", "daniel": "danieli",
        "esther": "esta", "ezekiel": "ezekieli", "philemon": "filemoni",
        "habakkuk": "habakuki", "haggai": "hagai", "numbers": "hesabu",
        "isaiah": "isaya", "deuteronomy": "kumbukumbu", "exodus": "kutoka",
        "luke": "luka", "malachi": "malaki", "lamentations": "maombolezo",
        "mark": "marko", "acts": "matendo", "matthew": "mathayo",
        "ecclesiastes": "mhubiri", "micah": "mika", "proverbs": "mithali",
        "genesis": "mwanzo", "nahum": "nahumu", "obadiah": "obadia",
        "ruth": "ruthu", "zephaniah": "sefania", "revelation": "ufunuo",
        "judges": "waamuzi", "hebrews": "waebrania", "ephesians": "waefeso",
        "philippians": "wafilipi", "galatians": "wagalatia",
        "colossians": "wakolosai", "leviticus": "walawi", "romans": "warumi",
        "songofsolomon": "wimbo", "james": "yakobo", "jeremiah": "yeremia",
        "joel": "yoeli", "john": "yohana", "jonah": "yona",
        "joshua": "yoshua", "jude": "yuda", "psalms": "zaburi",
        "zechariah": "zekaria",
    },
}

# Friendly spellings users type -> normalized English book id.
ENGLISH_ALIASES = {
    "psalm": "psalms",
    "song": "songofsolomon",
    "songofsongs": "songofsolomon",
    "revelations": "revelation",
    "canticles": "songofsolomon",
}

_book_map_cache: dict[str, dict[str, str]] = {}


def normalize_book(name: str) -> str:
    key = re.sub(r"[\s.\-]", "", name.lower())
    return ENGLISH_ALIASES.get(key, key)


def book_map(version: str) -> dict[str, str]:
    """English book id -> this version's local book slug."""
    if version in _book_map_cache:
        return _book_map_cache[version]
    mapping = dict(BOOK_OVERRIDES.get(version, {}))
    try:
        books = requests.get(f"{SCRIPTUREFLOW_BASE}/{version}/books.json",
                             timeout=TIMEOUT).json()
        for b in books:
            if b.get("canonical_book"):
                mapping[b["canonical_book"].replace("-", "")] = b["book_slug"]
    except requests.RequestException:
        pass  # offline: the built-in table still covers all 66 books
    _book_map_cache[version] = mapping
    return mapping


def parse_reference(ref: str):
    """'John 3:16' or '1 John 2:1-5' -> (book_id, chapter, verse, end_verse)."""
    m = re.match(
        r"^\s*([1-3]?\s*[^\d:]+?)\s*(\d+)\s*[:.]\s*(\d+)(?:\s*[-–]\s*(\d+))?\s*$",
        ref,
    )
    if not m:
        return None
    book = normalize_book(m.group(1))
    return book, int(m.group(2)), int(m.group(3)), (int(m.group(4)) if m.group(4) else None)


def clean_verse_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("¶", "")).strip()


def fetch_from_book_asset(version: str, book_slug: str, chapter: int,
                          verse: int, end_verse: int | None):
    """Fallback: fetch the whole book file and extract the verses ourselves.

    ScriptureFlow's /api/verse lookup 404s for book slugs containing
    non-ASCII letters (ɛ, ɔ, ʋ …) even though the book assets exist —
    confirmed July 6, 2026. The static /{version}/books/{slug}.json route
    works for all slugs, so we pull the verses out of it directly.
    """
    r = requests.get(f"{SCRIPTUREFLOW_BASE}/{version}/books/{book_slug}.json",
                     timeout=TIMEOUT)
    r.raise_for_status()
    book = r.json()
    chapters = {c["chapter"]: c for c in book.get("chapters", [])}
    if chapter not in chapters:
        raise ValueError(f"chapter {chapter} not found")
    verses = {v["verse"]: v["text"] for v in chapters[chapter].get("verses", [])}
    wanted = range(verse, (end_verse or verse) + 1)
    texts = [verses[n] for n in wanted if n in verses]
    if not texts:
        raise ValueError(f"verse {verse} not found in chapter {chapter}")
    name = book.get("translation_name") or version
    return clean_verse_text(" ".join(texts)), name


def fetch_passage(version: str, book_slug: str, chapter: int, verse: int,
                  end_verse: int | None):
    """Returns (text, translation_name) or raises ValueError with a friendly message."""
    params = {"version": version, "book": book_slug,
              "chapter": chapter, "verse": verse}
    if end_verse:
        params["end_verse"] = end_verse
    r = requests.get(f"{SCRIPTUREFLOW_BASE}/api/verse", params=params,
                     timeout=TIMEOUT)
    data = r.json()
    if not data.get("ok"):
        return fetch_from_book_asset(version, book_slug, chapter, verse,
                                     end_verse)
    if data.get("text"):
        text = data["text"]
    elif data.get("verses"):
        text = " ".join(v.get("text", "") for v in data["verses"])
    else:
        raise ValueError("no verse text in response")
    result = data.get("result") or {}
    if isinstance(result, list):  # passage ranges return a list of verses
        result = result[0] if result else {}
    name = result.get("translation_name") or version
    return clean_verse_text(text), name


# ---------------------------------------------------------------------------
# AI layer. Prompts are real; transport falls back to mocks until a backend
# is configured (MI300X endpoint or Fireworks key via environment variables).
# ---------------------------------------------------------------------------

DRAFT_PROMPT = """You are helping a ministry worker create Scripture study materials.

Below is a Bible passage in two parallel translations: English, and {language}
(an expert human translation). Write a {material} based on this passage in TWO
versions: first in {language}, then the same material in English. Both versions
must carry the same meaning.

Audience: {audience}. Shape vocabulary, examples, and tone for this audience.
{ministry_context}
Rules:
- Match the vocabulary, spelling, and register of the {language} translation.
- Quote the {language} translation exactly when quoting the passage.
- Do NOT translate or alter the Scripture text itself.
- In the {language} version, after the draft, list any phrases you are less
  confident about under the heading "NEEDS NATIVE-SPEAKER REVIEW", one per line.
- Format your answer EXACTLY like this, keeping the marker lines:

=== {language} VERSION ===
(the {material} in {language})

=== ENGLISH VERSION ===
(the same {material} in English)

Passage reference: {reference}

English ({english_name}):
{english_text}

{language} ({target_name}):
{target_text}

Write both versions now."""


def ministry_context(name: str, church: str, location: str,
                     audience_desc: str) -> str:
    parts = []
    if church.strip():
        parts.append(f"ministry: {church.strip()}")
    if name.strip():
        parts.append(f"prepared by: {name.strip()}")
    if location.strip():
        parts.append(f"location: {location.strip()}")
    if audience_desc.strip():
        parts.append(f"about the audience: {audience_desc.strip()}")
    if not parts:
        return ""
    return ("Context about this ministry (tailor tone and examples): "
            + "; ".join(parts) + ".\n")


def split_dual_draft(text: str, language: str) -> tuple[str, str]:
    """Split '=== X VERSION ===' sections into (target, english).

    If the model ignored the format, keep everything in the target box
    rather than losing output."""
    parts = re.split(r"===\s*([^=]+?)\s*VERSION\s*===", text)
    tgt, eng = "", ""
    for name, body in zip(parts[1::2], parts[2::2]):
        if name.strip().lower().startswith("english"):
            eng = body.strip()
        else:
            tgt = body.strip()
    if not tgt and not eng:
        return text.strip(), ""
    return tgt, eng

BACK_TRANSLATE_PROMPT = """Translate the following {language} text into plain English,
as literally as possible so the reader can verify the meaning. Do not improve
or embellish it — translate what it actually says.

{draft}"""


_resolved_models: dict[str, str] = {}


def list_models(base: str, key: str) -> list[str]:
    """Ask an OpenAI-compatible endpoint (Fireworks or vLLM) what it serves."""
    r = requests.get(f"{base}/v1/models",
                     headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


def pick_gemma_model(model_ids: list[str]) -> str | None:
    """Ladder: newest Gemma generation first, largest size first,
    instruct-tuned preferred (Gemma 4 27B > Gemma 4 12B > Gemma 3 27B …)."""
    def score(mid: str):
        low = mid.lower()
        gen = re.search(r"gemma-?(\d+)", low)
        size = re.search(r"(\d+(?:\.\d+)?)b", low)
        instruct = bool(re.search(r"(^|[^a-z])(it|instruct)([^a-z]|$)", low))
        return (instruct,
                int(gen.group(1)) if gen else 0,
                float(size.group(1)) if size else 0.0)
    gemmas = [m for m in model_ids if "gemma" in m.lower()]
    return max(gemmas, key=score) if gemmas else None


def resolve_model(base: str, key: str) -> str | None:
    """Model to use at this endpoint: env override, else discovered Gemma,
    else (for the self-hosted vLLM box) whatever single model it serves."""
    override = os.environ.get("GEMMA_MODEL")
    if override:
        return override
    if base in _resolved_models:
        return _resolved_models[base]
    try:
        models = list_models(base, key)
    except requests.RequestException:
        return None
    model = pick_gemma_model(models)
    if not model and base != FIREWORKS_BASE and models:
        model = models[0]
    if model:
        _resolved_models[base] = model
    return model


def ai_complete(prompt: str) -> tuple[str, str] | None:
    """Try real backends in order. Returns (text, backend_label) or None."""
    backends = []
    if mi300x_url():
        backends.append((mi300x_url(), "EMPTY", "Gemma on AMD MI300X (vLLM)"))
    if fireworks_key():
        backends.append((FIREWORKS_BASE, fireworks_key(),
                         "Gemma via Fireworks AI"))
    for base, key, label in backends:
        model = resolve_model(base, key)
        if not model:
            continue
        try:
            r = requests.post(
                f"{base}/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                },
                timeout=120,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            return text, f"{label} · {model.rsplit('/', 1)[-1]}"
        except (requests.RequestException, KeyError, IndexError):
            continue  # fall down the ladder
    return None


MOCK_BANNER = (
    "⚠️ MOCK MODE — placeholder output. Real Gemma responses "
    "arrive once the Fireworks API key or MI300X endpoint is configured.\n\n"
)


def mock_draft(material: str, language: str, reference: str,
               target_text: str, audience: str) -> str:
    body = {
        "Devotional": (
            f"[{language} devotional on {reference} for {audience} would "
            "appear here — an opening reflection quoting the passage, a "
            "real-life application for the community, and a closing prayer.]"
        ),
        "Discussion questions": (
            f"1. [{language} question about what the passage says]\n"
            f"2. [{language} question about what it means]\n"
            f"3. [{language} question about how to live it this week]\n"
            f"4. [{language} question for the group to discuss together]\n"
            f"5. [{language} question connecting the passage to daily life]"
        ),
        "Summary": (
            f"[{language} summary of {reference} would appear here — "
            "3–4 sentences in simple language restating the passage's "
            "main point for readers and oral learners.]"
        ),
    }[material]
    # Same '=== X VERSION ===' format the real prompt requires, so the
    # splitter is exercised in mock mode too.
    return (
        f"=== {language} VERSION ===\n"
        + MOCK_BANNER
        + f"# {material} — {reference} ({language} · {audience})\n\n"
        + f"> {target_text}\n\n"
        + body
        + "\n\nNEEDS NATIVE-SPEAKER REVIEW:\n"
        + "- [phrases Gemma is less confident about will be listed here]\n\n"
        + "=== ENGLISH VERSION ===\n"
        + MOCK_BANNER
        + f"# {material} — {reference} (English · {audience})\n\n"
        + f"[The same {material.lower()} in English would appear here.]"
    )


def mock_back_translation(language: str) -> str:
    return (
        MOCK_BANNER
        + "[The draft above, translated literally back into English so a "
        f"non-{language}-speaker can verify the meaning survived, will "
        "appear here.]"
    )


# ---------------------------------------------------------------------------
# UI handlers
# ---------------------------------------------------------------------------

def on_fetch(language: str, reference: str):
    version = LANGUAGES[language]
    parsed = parse_reference(reference or "")
    if not parsed:
        return "", "", (
            "⚠️ Couldn't read that reference. Try the form "
            "**John 3:16** or **James 1:2-4** — or paste the passage "
            "text into the boxes below by hand."
        )
    book, chapter, verse, end_verse = parsed
    tgt_map = book_map(version)
    eng_map = book_map(ENGLISH_VERSION)
    if book not in eng_map:
        return "", "", f"⚠️ Unknown book name in “{reference}”."
    try:
        eng_text, eng_name = fetch_passage(ENGLISH_VERSION, eng_map[book],
                                           chapter, verse, end_verse)
    except (requests.RequestException, ValueError) as e:
        return "", "", (
            f"⚠️ English fetch failed ({e}). You can paste the "
            "passage into the boxes by hand and still generate a draft."
        )
    if book not in tgt_map:
        return eng_text, "", (
            f"⚠️ That book isn't available in the {language} "
            "translation. Paste the passage by hand if you have it."
        )
    try:
        tgt_text, tgt_name = fetch_passage(version, tgt_map[book],
                                           chapter, verse, end_verse)
    except (requests.RequestException, ValueError) as e:
        return eng_text, "", (
            f"⚠️ {language} fetch failed ({e}). Paste the passage "
            "by hand if you have it."
        )
    return eng_text, tgt_text, (
        f"✅ Fetched **{reference.strip()}** · English: {eng_name} "
        f"· {language}: {tgt_name} · via ScriptureFlow"
    )


def on_draft(language: str, material: str, audience: str, reference: str,
             eng_text: str, tgt_text: str, min_name: str, min_church: str,
             min_location: str, min_audience: str):
    if not tgt_text.strip():
        return "", "", "⚠️ No passage text yet — fetch or paste one first."
    reference = reference.strip() or "the passage"
    prompt = DRAFT_PROMPT.format(
        language=language, material=material.lower(), audience=audience,
        ministry_context=ministry_context(min_name, min_church,
                                          min_location, min_audience),
        reference=reference,
        english_name=ENGLISH_VERSION, english_text=eng_text.strip() or "(not provided)",
        target_name=LANGUAGES[language], target_text=tgt_text.strip(),
    )
    real = ai_complete(prompt)
    if real:
        text, backend = real
        tgt, eng = split_dual_draft(text, language)
        return tgt, eng, f"✅ Draft generated by {backend}."
    tgt, eng = split_dual_draft(
        mock_draft(material, language, reference, tgt_text.strip(), audience),
        language)
    return tgt, eng, "⚠️ Draft is a MOCK — no AI backend configured yet."


def make_download(language: str, material: str, audience: str, reference: str,
                  eng_text: str, tgt_text: str, draft_tgt: str,
                  draft_eng: str, back_text: str, min_name: str,
                  min_church: str, min_location: str):
    reference = reference.strip() or "passage"
    lines = [f"# {material} — {reference} ({language})", ""]
    if min_church.strip():
        lines += [f"**Ministry:** {min_church.strip()}"]
    if min_name.strip():
        lines += [f"**Prepared by:** {min_name.strip()}"]
    if min_location.strip():
        lines += [f"**Location:** {min_location.strip()}"]
    lines += [f"**Audience:** {audience}", "", "## Passage", ""]
    if eng_text.strip():
        lines += [f"**English:** {eng_text.strip()}", ""]
    if tgt_text.strip():
        lines += [f"**{language}:** {tgt_text.strip()}", ""]
    if draft_tgt.strip():
        lines += [f"## {material} — {language}", "", draft_tgt.strip(), ""]
    if draft_eng.strip():
        lines += [f"## {material} — English", "", draft_eng.strip(), ""]
    if back_text.strip():
        lines += ["## Back-translation (verification)", "", back_text.strip(), ""]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines += ["---",
              f"Generated by Every Tongue · Scripture text via ScriptureFlow · {stamp}",
              "AI drafts require native-speaker review before use."]
    safe_ref = re.sub(r"[^A-Za-z0-9]+", "-", reference).strip("-").lower()
    path = os.path.join(tempfile.gettempdir(), f"every-tongue-{safe_ref}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def on_back_translate(language: str, draft: str):
    if not draft.strip():
        return "", "⚠️ Nothing to back-translate — generate a draft first."
    real = ai_complete(BACK_TRANSLATE_PROMPT.format(language=language, draft=draft))
    if real:
        text, backend = real
        return text, f"✅ Back-translation by {backend}."
    return (mock_back_translation(language),
            "⚠️ Back-translation is a MOCK — no AI backend configured yet.")


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="Every Tongue") as demo:
    gr.Markdown(
        "# \U0001f30d Every Tongue\n"
        "**AI-drafted Scripture study materials in low-resource languages** "
        "— grounded in expert human translations via ScriptureFlow, "
        "drafted by Gemma, checked by you."
    )

    with gr.Row():
        language = gr.Dropdown(list(LANGUAGES), value="Akuapem Twi",
                               label="Target language")
        material = gr.Dropdown(MATERIAL_TYPES, value="Devotional",
                               label="Material type")
        audience = gr.Dropdown(AUDIENCES, value="General Congregation",
                               label="Audience")
        reference = gr.Textbox(label="Passage reference",
                               placeholder="e.g. John 3:16 or James 1:2-4")

    with gr.Accordion("About your ministry (optional — shapes the drafts)",
                      open=False):
        with gr.Row():
            min_name = gr.Textbox(label="Your name")
            min_church = gr.Textbox(label="Church / ministry name")
            min_location = gr.Textbox(label="Location")
        min_audience = gr.Textbox(
            label="Describe your audience",
            placeholder="e.g. youth group in Kumasi, mixed reading levels, "
                        "many oral learners", lines=2)
        gr.Markdown("*Session only — nothing is saved after you close the page.*")

    fetch_btn = gr.Button("\U0001f4d6 Fetch passage", variant="primary")
    status = gr.Markdown()

    with gr.Row():
        eng_box = gr.Textbox(label="English passage (editable — paste by "
                                   "hand if the fetch fails)", lines=4)
        tgt_box = gr.Textbox(label="Target-language passage (editable)", lines=4)

    draft_btn = gr.Button("✍️ Draft study material", variant="primary")
    with gr.Row():
        draft_tgt_box = gr.Textbox(label="Draft — target language", lines=14)
        draft_eng_box = gr.Textbox(label="Draft — English", lines=14)

    with gr.Row():
        back_btn = gr.Button("\U0001f504 Back-translate draft to English")
        download_btn = gr.DownloadButton("⬇️ Download as Markdown")
    back_box = gr.Textbox(label="Back-translation (verify the meaning survived)",
                          lines=8)

    gr.Markdown(
        "---\nScripture text served by [ScriptureFlow]"
        "(https://scriptureflow-api-preview.pages.dev); translation names are "
        "shown with each fetch. AI drafts are starting points — always "
        "have a native speaker review before use."
    )

    fetch_btn.click(on_fetch, [language, reference], [eng_box, tgt_box, status])
    draft_btn.click(on_draft,
                    [language, material, audience, reference, eng_box, tgt_box,
                     min_name, min_church, min_location, min_audience],
                    [draft_tgt_box, draft_eng_box, status])
    back_btn.click(on_back_translate, [language, draft_tgt_box],
                   [back_box, status])
    download_btn.click(make_download,
                       [language, material, audience, reference, eng_box,
                        tgt_box, draft_tgt_box, draft_eng_box, back_box,
                        min_name, min_church, min_location],
                       download_btn)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
