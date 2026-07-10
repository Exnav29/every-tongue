"""Every Tongue — AI-drafted Scripture study materials in low-resource languages.

Data layer: ScriptureFlow (pre-existing open API) — parallel-text grounding:
the passage is fetched in BOTH English and the target language, and the pair
is sent to Gemma so drafts match the vocabulary of the expert translation.

AI layer (failure ladder): self-hosted Gemma on AMD MI300X via vLLM when the
endpoint is up -> Fireworks AI -> mock responses (no key configured yet).

Built for the AMD Developer Hackathon: ACT II, July 2026.
"""

import json
import os
import re
import tempfile
from datetime import datetime, timezone

import gradio as gr
import requests

# Hugging Face ZeroGPU (the free tier for Gradio Spaces) requires at least one
# @spaces.GPU function to be present at startup, or the Space is killed with
# "No @spaces.GPU function detected". Every Tongue runs its actual inference on
# EXTERNAL endpoints (Fireworks / self-hosted AMD Gemma), not on the Space, so
# this is a minimal never-called stub purely to satisfy that check. Guarded so
# local runs (without the `spaces` package) are unaffected.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_stub():
        return "ok"
except Exception:
    pass

SCRIPTUREFLOW_BASE = "https://scriptureflow-api-preview.pages.dev"
TIMEOUT = 20

# English source versions — all verified to return clean text (Tue Jul 7).
# BSB and FBV were rejected: they embed unbounded prose footnotes mid-verse
# that can't be stripped without risking Scripture text.
ENGLISH_VERSIONS = {
    "Translation for Translators (plain modern English)": "en-t4t",
    "Literal Standard Version": "en-lsv",
    "American Standard Version (1901)": "en-asv",
    "King James Version": "en-kjv",
}
DEFAULT_ENGLISH = "Translation for Translators (plain modern English)"

FIREWORKS_BASE = "https://api.fireworks.ai/inference"


# AI backends — configured via environment variables, never committed.
# Read at call time so a key pasted mid-session only needs an app restart,
# not a code change.
def fireworks_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY", "")
    if not key and os.name == "nt":
        # Windows: a user-level env var set after this process's parent
        # started isn't inherited — read it from the user environment
        # directly so a freshly pasted key works without a machine reboot.
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = str(winreg.QueryValueEx(k, "FIREWORKS_API_KEY")[0])
        except OSError:
            key = ""
    return key


def mi300x_url() -> str:
    return os.environ.get("MI300X_BASE_URL", "").rstrip("/")

# Target translations.  label -> (ScriptureFlow version key, language name).
#
# VERIFIED = hand-mapped, Latin-script, font-covered, end-to-end tested.
# Swahili is the demo hero (Gemma is fluent in it); Akuapem Twi stays as the
# showcased low-resource "hard case". The rest of ScriptureFlow's ~199
# translations are added dynamically as EXPERIMENTAL so a visitor sees the
# real breadth — with graceful degradation (reference lookup may be
# unavailable -> paste-in; PDF steers non-Latin/RTL scripts to Markdown).
VERIFIED_TARGETS = {
    "Swahili — Kiswahili Neno 2015": ("swh-onen", "Swahili"),
    "Akuapem Twi — Biblica Open 2020": ("tw-wakna", "Akuapem Twi"),
    "Asante Twi — Biblica Open 2020": ("tw-wasna", "Asante Twi"),
    "Ewe — eweOAL 2020": ("ee-oal", "Ewe"),
}
VERIFIED_VERSIONS = {v for v, _ in VERIFIED_TARGETS.values()}
DEFAULT_TARGET = "Swahili — Kiswahili Neno 2015"


def _load_language_names() -> dict:
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "language_names.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def build_target_translations() -> dict:
    """Verified languages first, then the full ScriptureFlow corpus as
    experimental. Falls back to verified-only if the catalog can't be
    fetched, so the app never breaks on a startup network hiccup."""
    mapping = dict(VERIFIED_TARGETS)
    names = _load_language_names()
    try:
        catalog = requests.get(f"{SCRIPTUREFLOW_BASE}/translations.json",
                               timeout=15).json()
    except Exception:
        return mapping  # verified-only
    experimental = []
    for t in catalog:
        if t.get("status") != "ready" or t.get("version") in VERIFIED_VERSIONS:
            continue
        ver = t["version"]
        name = names.get(t.get("language_code", ""), "") or t.get("language_code", "?").upper()
        tname = (t.get("translation_name") or "").strip()
        label = f"{name} — {tname}" if tname else name
        base, i = label, 2
        while label in mapping or label in {e[0] for e in experimental}:
            label, i = f"{base} ({i})", i + 1
        experimental.append((label, ver, name))
    experimental.sort(key=lambda e: e[0].lower())
    for label, ver, name in experimental:
        mapping[label] = (ver, name)
    return mapping


TARGET_TRANSLATIONS = build_target_translations()

MATERIAL_TYPES = ["Study guide", "Devotional", "Discussion questions", "Quick Read"]
DEFAULT_MATERIAL = "Study guide"

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
    # T4T uses square-bracket translator markers like [MTY], [MET], [RHQ].
    text = re.sub(r"\s*\[[A-Z]{2,6}\]", "", text)
    # KJV embeds translator footnotes inline, e.g.
    # "...saith the LORD.29.9 falsely: Heb. in a lie For thus saith..."
    # Pattern: chapter.verse, the flagged word, a colon, a source marker
    # (Heb./Gr./Chald./or/that is), then the note — which runs lowercase
    # until the next sentence's capital letter.
    text = re.sub(
        r"\s*\d+\.\d+\s+[^:]{1,50}:\s*(?:Heb|Gr|Chald|or|that is)\b\.?,?[^A-Z]*",
        " ", text)
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

# Each material type sets its own SECTION STRUCTURE. Audience is orthogonal:
# it sets tone/reading level, not which sections appear. Every structure uses
# Markdown "# "/"## " headings so the PDF/Markdown renderers stay generic; the
# review section is added to the target version only (see shared rules below).
MATERIAL_STRUCTURES = {
    "Study guide": """# (Title — a short line capturing the passage's main message)
## Passage
   Quote the relevant verse(s), one or two at a time.
## Background
   The historical, cultural, and authorial background of the passage.
## Immediate Context
   What comes just before and after, and how it frames this passage.
## Explanation
   A clear verse-by-verse or phrase-by-phrase explanation of the meaning.
## Key Spiritual Themes
   The main spiritual truths, as a short bulleted list.
## Modern-Day Application
   Concrete ways today's readers can live this out.
## Reflection Questions
   A NUMBERED list of 4-6 questions for personal or group study.
## Closing Prayer
   A short prayer flowing from the passage.""",

    "Devotional": """# (Title — a warm, inviting line)
## Passage
   Quote just one or two key verses.
## Reflection
   2-3 short, warm paragraphs connecting the verse to everyday life. Personal
   and encouraging, not academic.
## One Thought to Carry
   A single sentence the reader can hold onto today.
## Closing Prayer
   A short, heartfelt prayer.""",

    "Discussion questions": """# (Title — a short line naming the theme)
## Passage
   Quote the relevant verse(s).
## Context in Brief
   2-3 sentences of grounding so a group understands the setting.
## Observe
   A NUMBERED list of 2-3 questions about what the passage SAYS.
## Interpret
   A NUMBERED list of 2-3 questions about what it MEANS.
## Apply
   A NUMBERED list of 2-3 questions about how to LIVE it out.
## Closing Prayer
   A one- or two-sentence prayer prompt to close the discussion.""",

    "Quick Read": """# (Title — a short, clear line)
## Passage
   Quote the key verse(s).
## Summary
   3-5 plain, simple sentences restating the passage's main point — easy to
   read aloud for oral learners.
## Key Points
   A short bulleted list of the main takeaways.
## One-Line Takeaway
   A single memorable sentence.""",
}

DRAFT_PROMPT = """You are an experienced Bible teacher helping a ministry worker
create Scripture study material. The material type is: {material}.

Below is a Bible passage in two parallel translations: English, and {language}
(an expert human translation). Produce the material in TWO versions: first in
{language}, then the same material in English. Both versions must carry the
same meaning and structure.

The MATERIAL TYPE sets the structure (the sections below); the AUDIENCE sets
the tone and reading level. They are independent — e.g. a Devotional for
Children or a Study guide for Seniors are both valid. Audience: {audience} —
shape vocabulary, examples, depth, and reading level for them.
{ministry_context}
Write each version with EXACTLY these parts, in this order. Use a Markdown
"# " line for the title and a Markdown "## " heading for every section. In the
{language} version, translate the section headings into {language}; in the
English version, keep them in English.

{structure}

Rules:
- Match the vocabulary, spelling, and register of the {language} translation.
- In the {language} version quote ONLY the {language} translation; in the
  English version quote ONLY the English translation. Never quote the passage
  in the other language's version.
- Write the English version entirely in English and the {language} version
  entirely in {language} (proper names excepted). Do not drop untranslated
  English words into the {language} version.
- Do NOT translate or alter the Scripture text itself.
- After the final section of the {language} version ONLY, add one more
  "## NEEDS NATIVE-SPEAKER REVIEW" section listing any {language} words or
  phrases you are less confident about, one per line.
- Format your whole answer EXACTLY like this, keeping the marker lines:

=== {language} VERSION ===
(the full material in {language})

=== ENGLISH VERSION ===
(the same material in English)

Passage reference: {reference}

English ({english_name}):
{english_text}

{language} ({target_name}):
{target_text}

Write both versions now."""


def ministry_context(name: str, church: str, location: str,
                     audience_desc: str) -> str:
    # Gradio passes None (not "") for fields inside a never-opened accordion.
    name, church, location, audience_desc = (
        (v or "") for v in (name, church, location, audience_desc))
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


def fireworks_gemma_catalog(key: str) -> list[str]:
    """Fireworks' /v1/models lists only ~7 featured models; the Gemma family
    lives in the full public catalog, which is paginated on the control-plane
    API (confirmed Tue Jul 7)."""
    url = "https://api.fireworks.ai/v1/accounts/fireworks/models"
    names: list[str] = []
    token = None
    for _ in range(20):
        params: dict = {"pageSize": 200}
        if token:
            params["pageToken"] = token
        r = requests.get(url, headers={"Authorization": f"Bearer {key}"},
                         params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for m in data.get("models", []):
            name = m.get("name", "")
            if "gemma" in name.lower() and m.get("state") == "READY":
                names.append("accounts/fireworks/models/" + name.split("/")[-1])
        token = data.get("nextPageToken")
        if not token:
            break
    return names


QUANT_MARKERS = ("nvfp4", "fp8", "awq", "gptq", "int4", "int8")

# TEMPORARY STAND-IN (added Tue Jul 7). As of today NO Gemma model is
# serverless-deployed on Fireworks (all 15 catalog Gemmas have zero
# deployments; calls return 404 "not deployed"). Until the organizers'
# promised Gemma route goes live, fall back to the strongest currently
# deployed model so prompt/format tuning can happen on real inference.
# Gemma stays at the TOP of the ladder: the moment Fireworks deploys one,
# it wins the probe and this stand-in is ignored automatically.
FIREWORKS_STANDIN = "accounts/fireworks/models/gpt-oss-120b"


def rank_gemma_models(model_ids: list[str]) -> list[str]:
    """Ladder: instruct-tuned first, newest Gemma generation first,
    full-precision before quantized variants, then largest size
    (Gemma 4 31B > Gemma 4 26B > Gemma 3 27B > … ), best first."""
    def score(mid: str):
        low = mid.lower()
        # Generation digit must NOT be a size ("gemma-7b" is size 7B,
        # generation 1 — not generation 7).
        gen = re.search(r"gemma-?(\d+)(?=$|-)", low)
        size = re.search(r"(\d+(?:\.\d+)?)b", low)
        instruct = bool(re.search(r"(^|[^a-z])(it|instruct)([^a-z]|$)", low))
        quantized = any(q in low for q in QUANT_MARKERS)
        return (instruct,
                int(gen.group(1)) if gen else 0,
                not quantized,
                float(size.group(1)) if size else 0.0)
    gemmas = [m for m in model_ids if "gemma" in m.lower()]
    return sorted(gemmas, key=score, reverse=True)


def pick_gemma_model(model_ids: list[str]) -> str | None:
    ranked = rank_gemma_models(model_ids)
    return ranked[0] if ranked else None


def probe_model(base: str, key: str, model: str) -> bool:
    """One-token test call — catalog listings include models that aren't
    actually callable (404), so verify before trusting a pick."""
    try:
        r = requests.post(
            f"{base}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model,
                  "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 1},
            timeout=30,
        )
        return r.ok
    except requests.RequestException:
        return False


def resolve_model(base: str, key: str) -> str | None:
    """Model to use at this endpoint: env override, else the best Gemma
    that actually answers a probe, else (for the self-hosted vLLM box)
    whatever single model it serves."""
    override = os.environ.get("GEMMA_MODEL")
    if override:
        return override
    if base in _resolved_models:
        return _resolved_models[base]
    try:
        if base == FIREWORKS_BASE:
            models = fireworks_gemma_catalog(key)
        else:
            models = list_models(base, key)
    except requests.RequestException:
        return None
    candidates = rank_gemma_models(models)[:6]
    if base == FIREWORKS_BASE:
        candidates.append(FIREWORKS_STANDIN)  # temporary — see note above
    elif not candidates and models:
        candidates = models[:1]
    for model in candidates:  # walk the ladder until one answers
        if probe_model(base, key, model):
            _resolved_models[base] = model
            return model
    return None


def ai_complete(prompt: str) -> tuple[str, str, bool] | None:
    """Try real backends in order.
    Returns (text, backend_label, truncated) or None."""
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
                    # Dual-language drafts of long passages need room; 2048
                    # truncated the English half (seen Tue Jul 7).
                    "max_tokens": 8192,
                },
                timeout=180,
            )
            r.raise_for_status()
            choice = r.json()["choices"][0]
            text = choice["message"]["content"]
            truncated = choice.get("finish_reason") == "length"
            short = model.rsplit("/", 1)[-1]
            if "gemma" not in model.lower():
                label = "TEMPORARY stand-in (not Gemma) via Fireworks AI"
            return text, f"{label} · {short}", truncated
        except (requests.RequestException, KeyError, IndexError):
            continue  # fall down the ladder
    return None


MOCK_BANNER = (
    "⚠️ MOCK MODE — placeholder output. Real Gemma responses "
    "arrive once the Fireworks API key or MI300X endpoint is configured.\n\n"
)


def _mock_guide(reference: str, material: str, audience: str, label: str,
                passage: str) -> str:
    """Build a mock version from the selected material type's own structure —
    so mock output matches the real per-type sections and exercises the
    splitter and PDF/Markdown renderers for every type."""
    spec = MATERIAL_STRUCTURES.get(material, MATERIAL_STRUCTURES["Study guide"])
    out = [f"# {material} on {reference} ({label} · {audience})", ""]
    for ln in spec.splitlines():
        s = ln.strip()
        if not s.startswith("## "):
            continue
        heading = s[3:].strip()
        low = heading.lower()
        out.append(f"## {heading}")
        if low == "passage":
            out.append(f"> {passage}")
        elif any(w in low for w in ("question", "observe", "interpret", "apply")):
            out += [f"1. [{label} {low} question]", f"2. [{label} {low} question]"]
        elif "key" in low or "theme" in low or "point" in low:
            out += ["- [point one]", "- [point two]"]
        else:
            out.append(f"[{label} {low} for {audience.lower()} would appear here.]")
        out.append("")
    return "\n".join(out)


def mock_draft(material: str, language: str, reference: str,
               target_text: str, audience: str) -> str:
    # Same '=== X VERSION ===' + section format the real prompt requires, so
    # the splitter and PDF renderer are exercised in mock mode too.
    return (
        f"=== {language} VERSION ===\n"
        + MOCK_BANNER
        + _mock_guide(reference, material, audience, language, target_text)
        + "\n## NEEDS NATIVE-SPEAKER REVIEW\n"
        + "- [phrases Gemma is less confident about will be listed here]\n\n"
        + "=== ENGLISH VERSION ===\n"
        + MOCK_BANNER
        + _mock_guide(reference, material, audience, "English",
                      "[the passage in English]")
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

def on_fetch(target_label: str, english_label: str, reference: str):
    version, language = TARGET_TRANSLATIONS[target_label]
    english_version = ENGLISH_VERSIONS[english_label]
    parsed = parse_reference(reference or "")
    if not parsed:
        return "", "", (
            "⚠️ Couldn't read that reference. Try the form "
            "**John 3:16** or **James 1:2-4** — or paste the passage "
            "text into the boxes below by hand."
        )
    book, chapter, verse, end_verse = parsed
    tgt_map = book_map(version)
    eng_map = book_map(english_version)
    if book not in eng_map:
        return "", "", f"⚠️ Unknown book name in “{reference}”."
    try:
        eng_text, eng_name = fetch_passage(english_version, eng_map[book],
                                           chapter, verse, end_verse)
    except (requests.RequestException, ValueError) as e:
        return "", "", (
            f"⚠️ English fetch failed ({e}). You can paste the "
            "passage into the boxes by hand and still generate a draft."
        )
    if book not in tgt_map:
        verified = version in VERIFIED_VERSIONS
        note = ("" if verified else
                " (reference lookup isn't wired up for this experimental "
                "translation yet)")
        return eng_text, "", (
            f"ℹ️ Couldn't auto-fetch this book in {language}{note} — the "
            "English side is loaded; **paste the "
            f"{language} passage** into the box on the right to continue."
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


def on_draft(target_label: str, english_label: str, material: str,
             audience: str, reference: str, eng_text: str, tgt_text: str,
             min_name: str, min_church: str, min_location: str,
             min_audience: str):
    version, language = TARGET_TRANSLATIONS[target_label]
    if not (tgt_text or "").strip():
        return "", "", "⚠️ No passage text yet — fetch or paste one first."
    eng_text = eng_text or ""
    reference = (reference or "").strip() or "the passage"
    structure = MATERIAL_STRUCTURES.get(material, MATERIAL_STRUCTURES["Study guide"])
    prompt = DRAFT_PROMPT.format(
        language=language, material=material, audience=audience,
        structure=structure,
        ministry_context=ministry_context(min_name, min_church,
                                          min_location, min_audience),
        reference=reference,
        english_name=english_label, english_text=eng_text.strip() or "(not provided)",
        target_name=target_label, target_text=tgt_text.strip(),
    )
    real = ai_complete(prompt)
    if real:
        text, backend, truncated = real
        tgt, eng = split_dual_draft(text, language)
        status = f"✅ Draft generated by {backend}."
        if truncated:
            status += (" ⚠️ Output hit the length limit and was cut off — "
                       "try a shorter passage or regenerate.")
        elif not eng:
            status += (" ⚠️ The model skipped the English version — "
                       "regenerate to get both.")
        return tgt, eng, status
    tgt, eng = split_dual_draft(
        mock_draft(material, language, reference, tgt_text.strip(), audience),
        language)
    return tgt, eng, "⚠️ Draft is a MOCK — no AI backend configured yet."


# ---------------------------------------------------------------------------
# Exports — Markdown (simple, complete) and PDF (polished ministry handout).
#
# The PDF layout mirrors the ScriptureFlow study-guide template (colors and
# structure extracted from the reference PDF, Tue Jul 7): text header with
# wordmark + eyebrow, document title, 2x2 metadata grid, teal section
# headings, cream passage box, footer rule with "Powered by ScriptureFlow".
# ---------------------------------------------------------------------------

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
# Palette lifted from the ScriptureFlow template's own content streams.
BRAND_TEAL = "#022f2e"    # wordmark + section headings
BRAND_ACCENT = "#0f766e"  # eyebrow / accents
BRAND_TITLE = "#0f172a"   # document title
BRAND_BODY = "#1f2933"    # body text
BRAND_MUTED = "#374151"   # metadata + footer
BRAND_RULE = "#d1d5db"    # hairline rules
BRAND_SOFT = "#f8f7f2"    # passage box background
_fonts_ready = False


def _ensure_fonts():
    """Register the bundled DejaVu fonts once (they cover Twi/Ewe glyphs)."""
    global _fonts_ready
    if _fonts_ready:
        return
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont as RLTTFont
    pdfmetrics.registerFont(RLTTFont("ET", os.path.join(FONTS_DIR, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(RLTTFont("ET-Bold", os.path.join(FONTS_DIR, "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("ET", normal="ET", bold="ET-Bold",
                                  italic="ET", boldItalic="ET-Bold")
    _fonts_ready = True


_dejavu_cmap = None


def _font_cmap():
    global _dejavu_cmap
    if _dejavu_cmap is None:
        from reportlab.pdfbase.ttfonts import TTFont as RLTTFont
        f = RLTTFont("cov", os.path.join(FONTS_DIR, "DejaVuSans.ttf"))
        _dejavu_cmap = f.face.charToGlyph
    return _dejavu_cmap


def pdf_unsupported_script(text: str) -> bool:
    """True when the PDF renderer can't do this text justice — so we steer to
    Markdown instead of producing a broken/tofu PDF. Covers two cases:
    right-to-left / complex-shaping scripts (Arabic, Hebrew — reportlab does
    no bidi or shaping), and scripts the bundled DejaVu font lacks glyphs for
    (Devanagari, Bengali, Tamil, Telugu, Thai, CJK …). Latin, Cyrillic, Greek,
    and Twi/Ewe pass through fine."""
    cmap = _font_cmap()
    for c in text:
        if 0x0590 <= ord(c) <= 0x08FF:  # Hebrew/Arabic/Syriac/Thaana — RTL
            return True
    exotic = [c for c in text if c.isalpha() and ord(c) >= 0x0500]
    if exotic:
        missing = sum(1 for c in exotic if ord(c) not in cmap)
        if missing > len(exotic) * 0.2:
            return True
    return False


def _safe_name(reference: str, ext: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "-", reference).strip("-").lower() or "passage"
    return os.path.join(tempfile.gettempdir(), f"every-tongue-{safe}.{ext}")


def _clean_all(*vals):
    return [(v or "").strip() for v in vals]


def build_markdown(language, material, audience, reference, eng_text, tgt_text,
                   draft_tgt, draft_eng, back_text, min_name, min_church,
                   min_location):
    (eng_text, tgt_text, draft_tgt, draft_eng, back_text, min_name,
     min_church, min_location) = _clean_all(
        eng_text, tgt_text, draft_tgt, draft_eng, back_text, min_name,
        min_church, min_location)
    reference = (reference or "").strip() or "passage"
    lines = ["**Every Tongue Reviewer Handout** — generated study material for "
             "review, not Scripture translation.", "",
             f"# {material} — {reference} ({language})", ""]
    if min_church:
        lines.append(f"**Ministry:** {min_church}")
    if min_name:
        lines.append(f"**Prepared by:** {min_name}")
    if min_location:
        lines.append(f"**Location:** {min_location}")
    lines += [f"**Audience:** {audience}", "", "## Passage", ""]
    if eng_text:
        lines += [f"**English:** {eng_text}", ""]
    if tgt_text:
        lines += [f"**{language}:** {tgt_text}", ""]
    if draft_tgt:
        lines += [f"## Study guide — {language}", "", draft_tgt, ""]
    if draft_eng:
        lines += ["## Study guide — English", "", draft_eng, ""]
    if back_text:
        lines += ["## Back-translation (verification)", "", back_text, ""]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines += ["---",
              f"Generated by Every Tongue · Scripture text via ScriptureFlow · {stamp}",
              "**Native-speaker and ministry-leader review required before "
              "teaching, publication, or distribution.**"]
    path = _safe_name(reference, "md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _inline(text: str) -> str:
    """Escape XML and convert **bold** for reportlab Paragraph markup."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def _guide_flowables(draft_text, styles):
    """Convert one study guide's markdown into reportlab flowables."""
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    flow = []
    lines = draft_text.splitlines()
    i = 0
    review_mode = False
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("# ") and not stripped.startswith("## "):
            flow.append(Paragraph(_inline(stripped[2:]), styles["title"]))
            flow.append(Spacer(1, 4 * mm))
        elif stripped.startswith("## "):
            heading = stripped[3:].strip()
            review_mode = "REVIEW" in heading.upper()
            style = styles["review_h"] if review_mode else styles["heading"]
            flow.append(Paragraph(_inline(heading), style))
        elif stripped.startswith(">"):
            # Passage quote — collect consecutive quote lines into a box.
            quote = [stripped.lstrip("> ").strip()]
            while i + 1 < len(lines) and lines[i + 1].strip().startswith(">"):
                i += 1
                quote.append(lines[i].strip().lstrip("> ").strip())
            para = Paragraph(_inline(" ".join(quote)), styles["quote"])
            box = Table([[para]], colWidths=[165 * mm])
            box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(BRAND_SOFT)),
                ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(BRAND_ACCENT)),
                ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]))
            flow.append(box)
            flow.append(Spacer(1, 3 * mm))
        elif re.match(r"^\d+[.)]\s", stripped):
            items = []
            while i < len(lines) and re.match(r"^\d+[.)]\s", lines[i].strip()):
                items.append(ListItem(Paragraph(
                    _inline(re.sub(r"^\d+[.)]\s", "", lines[i].strip())),
                    styles["body"])))
                i += 1
            flow.append(ListFlowable(items, bulletType="1", leftIndent=10 * mm))
            flow.append(Spacer(1, 2 * mm))
            continue
        elif stripped.startswith(("- ", "* ")):
            items = []
            while i < len(lines) and lines[i].strip().startswith(("- ", "* ")):
                body = lines[i].strip()[2:]
                st = styles["review_item"] if review_mode else styles["body"]
                items.append(ListItem(Paragraph(_inline(body), st)))
                i += 1
            flow.append(ListFlowable(items, bulletType="bullet", leftIndent=10 * mm))
            flow.append(Spacer(1, 2 * mm))
            continue
        else:
            flow.append(Paragraph(_inline(stripped), styles["body"]))
        i += 1
    return flow


def _pdf_styles():
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib import colors
    return {
        "wordmark": ParagraphStyle("wm", fontName="ET-Bold", fontSize=19,
                                   textColor=colors.HexColor(BRAND_TEAL), leading=22),
        "eyebrow": ParagraphStyle("eb", fontName="ET-Bold", fontSize=8.5,
                                  textColor=colors.HexColor(BRAND_ACCENT), leading=12,
                                  spaceAfter=10),
        "title": ParagraphStyle("t", fontName="ET-Bold", fontSize=19,
                                textColor=colors.HexColor(BRAND_TITLE), leading=23, spaceAfter=4),
        "meta": ParagraphStyle("m", fontName="ET", fontSize=9.5,
                               textColor=colors.HexColor(BRAND_MUTED), leading=15),
        "heading": ParagraphStyle("h", fontName="ET-Bold", fontSize=13,
                                  textColor=colors.HexColor(BRAND_TEAL), spaceBefore=11,
                                  spaceAfter=4, leading=16),
        "review_h": ParagraphStyle("rh", fontName="ET-Bold", fontSize=13,
                                   textColor=colors.HexColor(BRAND_ACCENT), spaceBefore=11,
                                   spaceAfter=4, leading=16),
        "body": ParagraphStyle("b", fontName="ET", fontSize=10.5, leading=15,
                               textColor=colors.HexColor(BRAND_BODY),
                               alignment=TA_JUSTIFY, spaceAfter=5),
        "quote": ParagraphStyle("q", fontName="ET", fontSize=10.5,
                               textColor=colors.HexColor(BRAND_TEAL), leading=15),
        "review_item": ParagraphStyle("ri", fontName="ET", fontSize=10,
                                      textColor=colors.HexColor(BRAND_MUTED), leading=13),
    }


class _NumberedCanvas:
    """Two-pass canvas so the footer can show 'Page N of M' like the template."""

    @staticmethod
    def make(footer_left):
        from reportlab.pdfgen import canvas as _canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib import colors

        class C(_canvas.Canvas):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self._saved = []

            def showPage(self):
                self._saved.append(dict(self.__dict__))
                self._startPage()

            def save(self):
                total = len(self._saved)
                for state in self._saved:
                    self.__dict__.update(state)
                    self._draw_footer(total)
                    super().showPage()
                super().save()

            def _draw_footer(self, total):
                w = A4[0]
                self.setStrokeColor(colors.HexColor(BRAND_RULE))
                self.setLineWidth(0.5)
                self.line(18 * mm, 15 * mm, w - 18 * mm, 15 * mm)
                self.setFillColor(colors.HexColor(BRAND_MUTED))
                self.setFont("ET", 8)
                self.drawString(18 * mm, 11 * mm, footer_left)
                self.drawRightString(
                    w - 18 * mm, 11 * mm,
                    f"Powered by ScriptureFlow · Page {self.getPageNumber()} of {total}")
                self.setFont("ET", 6.5)
                self.drawCentredString(
                    w / 2, 7 * mm,
                    "Native-speaker and ministry-leader review required before "
                    "teaching, publication, or distribution.")
        return C


def _render_guide_pdf(path, doc_language, translation_name, material, audience,
                      reference, draft_text, min_name, min_church, min_location):
    """Render one study-guide version as a polished PDF handout matching the
    ScriptureFlow template. Language-agnostic — used for both the target
    language and the English version."""
    _ensure_fonts()
    from reportlab.platypus import SimpleDocTemplate, Spacer, Paragraph, Table, TableStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors

    draft_text, min_name, min_church, min_location = _clean_all(
        draft_text, min_name, min_church, min_location)
    reference = (reference or "").strip() or "passage"
    styles = _pdf_styles()

    # The first '# Title' line in the draft becomes the document title; strip
    # it so it isn't rendered twice. Scan lines (not just position 0) so a
    # leading mock banner or blank line doesn't hide it.
    doc_title = f"{material} — {reference}"
    lines = draft_text.splitlines()
    for k, ln in enumerate(lines):
        s = ln.lstrip()
        if s.startswith("# ") and not s.startswith("## "):
            doc_title = s[2:].strip()
            del lines[k]
            break
    body_text = "\n".join(lines)

    footer_left = min_church or "Every Tongue"
    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=18 * mm,
                            bottomMargin=20 * mm, leftMargin=18 * mm,
                            rightMargin=18 * mm, title=doc_title)

    def meta_cell(label, value):
        return Paragraph(f"<b>{_inline(label)}:</b> {_inline(value or '—')}", styles["meta"])

    story = [
        Paragraph("Every Tongue", styles["wordmark"]),
        Paragraph("REVIEWER HANDOUT · MULTILINGUAL BIBLE STUDY GUIDE", styles["eyebrow"]),
        Paragraph(_inline(doc_title), styles["title"]),
        Paragraph("Generated study material for review — not Scripture translation.",
                  styles["meta"]),
        Spacer(1, 2 * mm),
    ]
    grid = Table(
        [[meta_cell("Passage", reference), meta_cell("Translation", translation_name)],
         [meta_cell("Language", doc_language), meta_cell("Prepared for", min_name)]],
        colWidths=[87 * mm, 87 * mm])
    grid.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.HexColor(BRAND_RULE)),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor(BRAND_RULE)),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (0, -1), 0),
    ]))
    story += [grid, Spacer(1, 5 * mm)]
    story += _guide_flowables(body_text or "(no study guide generated yet)", styles)

    doc.build(story, canvasmaker=_NumberedCanvas.make(footer_left))
    return path


def build_pdf(target_label, material, audience, reference, tgt_text,
              draft_tgt, min_name, min_church, min_location):
    """Target-language study-guide PDF."""
    _, language = TARGET_TRANSLATIONS.get(target_label, (None, target_label))
    translation_name = (target_label.split("—")[-1].strip()
                        if "—" in target_label else target_label)
    return _render_guide_pdf(
        _safe_name((reference or "passage"), "pdf"), language, translation_name,
        material, audience, reference, draft_tgt, min_name, min_church, min_location)


def build_pdf_english(english_label, material, audience, reference, draft_eng,
                      min_name, min_church, min_location):
    """English-version study-guide PDF (own file so both can be downloaded)."""
    return _render_guide_pdf(
        _safe_name((reference or "passage") + "-english", "pdf"), "English",
        english_label, material, audience, reference, draft_eng,
        min_name, min_church, min_location)


def on_back_translate(target_label: str, draft: str):
    _, language = TARGET_TRANSLATIONS[target_label]
    if not draft.strip():
        return "", "⚠️ Nothing to back-translate — generate a draft first."
    real = ai_complete(BACK_TRANSLATE_PROMPT.format(language=language, draft=draft))
    if real:
        text, backend, truncated = real
        status = f"✅ Back-translation by {backend}."
        if truncated:
            status += " ⚠️ Output was cut off at the length limit."
        return text, status
    return (mock_back_translation(language),
            "⚠️ Back-translation is a MOCK — no AI backend configured yet.")


def on_export_pdf(target_label, english_label, material, audience, reference,
                  eng_text, tgt_text, draft_tgt, draft_eng, back_text,
                  min_name, min_church, min_location):
    if not (draft_tgt or "").strip():
        return gr.update(visible=False), "⚠️ Generate a draft before exporting."
    _, language = TARGET_TRANSLATIONS.get(target_label, (None, target_label))
    if pdf_unsupported_script(draft_tgt):
        return gr.update(visible=False), (
            f"ℹ️ PDF export doesn't support the {language} script yet "
            "(right-to-left or complex scripts don't render correctly in the "
            "PDF). Use **Export as Markdown** instead — it preserves the text "
            "perfectly and opens anywhere.")
    try:
        path = build_pdf(target_label, material, audience, reference, tgt_text,
                         draft_tgt, min_name, min_church, min_location)
    except Exception as e:  # never crash the UI on an export problem
        return gr.update(visible=False), f"⚠️ PDF export failed: {e}"
    return (gr.update(value=path, visible=True),
            f"✅ {language} PDF ready — click to download.")


def on_export_pdf_english(target_label, english_label, material, audience,
                          reference, eng_text, tgt_text, draft_tgt, draft_eng,
                          back_text, min_name, min_church, min_location):
    if not (draft_eng or "").strip():
        return gr.update(visible=False), (
            "⚠️ No English draft yet — regenerate; the model may have skipped "
            "the English version.")
    try:
        path = build_pdf_english(english_label, material, audience, reference,
                                 draft_eng, min_name, min_church, min_location)
    except Exception as e:
        return gr.update(visible=False), f"⚠️ English PDF export failed: {e}"
    return gr.update(value=path, visible=True), "✅ English PDF ready — click to download."


def on_export_md(target_label, english_label, material, audience, reference,
                 eng_text, tgt_text, draft_tgt, draft_eng, back_text,
                 min_name, min_church, min_location):
    _, language = TARGET_TRANSLATIONS[target_label]
    if not (draft_tgt or "").strip():
        return gr.update(visible=False), "⚠️ Generate a draft before exporting."
    path = build_markdown(language, material, audience, reference, eng_text,
                          tgt_text, draft_tgt, draft_eng, back_text, min_name,
                          min_church, min_location)
    return gr.update(value=path, visible=True), "✅ Markdown ready — click to download."


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
    gr.Markdown(
        "### Grounded study-material drafting — **not Bible translation.** "
        "Powered by **Gemma on AMD GPU** "
        "([see evidence](https://github.com/Exnav29/every-tongue/tree/main/evidence/amd-gpu))."
    )
    gr.Markdown(
        "> ⚡ **Gemma on AMD** — Every Tongue runs the open **Gemma** model on "
        "**AMD GPU hardware** (AMD Radeon · ROCm · vLLM). "
        "[See the AMD benchmark & samples ↗]"
        "(https://github.com/Exnav29/every-tongue/tree/main/evidence/amd-gpu)"
    )
    gr.Markdown(
        "**Backend:** the live demo uses a **Fireworks (AMD-backed) stand-in "
        "model** — labeled honestly on each generation below — while the real "
        "**Gemma-on-AMD** run is documented in "
        "[repo evidence](https://github.com/Exnav29/every-tongue/tree/main/evidence/amd-gpu). "
        "**Scripture source:** ScriptureFlow. Every draft is a starting point — "
        "**native-speaker review required.** *(With no backend configured, the "
        "app clearly labels its output as mock preview.)*"
    )

    with gr.Row():
        language = gr.Dropdown(list(TARGET_TRANSLATIONS), value=DEFAULT_TARGET,
                               label="Target translation")
        english = gr.Dropdown(list(ENGLISH_VERSIONS), value=DEFAULT_ENGLISH,
                              label="English source translation")
    gr.Markdown(
        f"*The first 4 target translations are **verified** end-to-end. The "
        f"other {max(len(TARGET_TRANSLATIONS) - len(VERIFIED_TARGETS), 0)} are "
        "**ScriptureFlow's full corpus (experimental)** — generation quality "
        "varies (the low-resource languages are exactly the hard case this "
        "tool exists for), reference lookup may fall back to paste-in, and "
        "PDF export steers non-Latin/right-to-left scripts to Markdown.*"
    )
    with gr.Row():
        material = gr.Dropdown(MATERIAL_TYPES, value=DEFAULT_MATERIAL,
                               label="Material type")
        audience = gr.Dropdown(AUDIENCES, value="General Congregation",
                               label="Audience")
        reference = gr.Textbox(label="Passage reference",
                               placeholder="e.g. John 3:16 or James 1:2-4")
    gr.Markdown(
        "**Judge demo path:**  (1) **Swahili** · reference **John 3:16-17** → "
        "the strong-case output.  (2) **Akuapem Twi** · same reference → the "
        "low-resource stress test that shows why the grounding + "
        "back-translation + native-speaker-review guardrails matter. Pick any "
        "Material type / Audience."
    )

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
        draft_tgt_box = gr.Textbox(label="Target-Language Study Material", lines=14)
        draft_eng_box = gr.Textbox(label="English Version", lines=14)

    back_btn = gr.Button("\U0001f504 Back-translate draft to English")
    back_box = gr.Textbox(label="English Back-Translation (verify the meaning survived)",
                          lines=8)

    gr.Markdown("### Export")
    with gr.Row():
        pdf_btn = gr.Button("\U0001f4c4 Export target-language PDF", variant="primary")
        pdf_en_btn = gr.Button("\U0001f4c4 Export English PDF")
        md_btn = gr.Button("\U0001f4dd Export as Markdown")
    pdf_file = gr.File(label="Target-language PDF", visible=False, interactive=False)
    pdf_en_file = gr.File(label="English PDF", visible=False, interactive=False)
    md_file = gr.File(label="Markdown (both versions)", visible=False, interactive=False)

    gr.Markdown(
        "---\nScripture text served by [ScriptureFlow]"
        "(https://scriptureflow-api-preview.pages.dev); translation names are "
        "shown with each fetch. AI drafts are starting points — always "
        "have a native speaker review before use."
    )

    fetch_btn.click(on_fetch, [language, english, reference],
                    [eng_box, tgt_box, status])
    draft_btn.click(on_draft,
                    [language, english, material, audience, reference,
                     eng_box, tgt_box,
                     min_name, min_church, min_location, min_audience],
                    [draft_tgt_box, draft_eng_box, status])
    back_btn.click(on_back_translate, [language, draft_tgt_box],
                   [back_box, status])
    export_inputs = [language, english, material, audience, reference,
                     eng_box, tgt_box, draft_tgt_box, draft_eng_box, back_box,
                     min_name, min_church, min_location]
    pdf_btn.click(on_export_pdf, export_inputs, [pdf_file, status])
    pdf_en_btn.click(on_export_pdf_english, export_inputs, [pdf_en_file, status])
    md_btn.click(on_export_md, export_inputs, [md_file, status])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
