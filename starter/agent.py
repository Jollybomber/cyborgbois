from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path

try:
    from local_embeddings import DEFAULT_INDEX, DEFAULT_MODEL, DEFAULT_MODEL_DIR, LocalEmbeddingIndex
except Exception:  # local dense retrieval is optional
    DEFAULT_INDEX = Path("data/catalog_bge_embeddings.npz")
    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    DEFAULT_MODEL_DIR = Path("data/bge-small-en-v1.5")
    LocalEmbeddingIndex = None


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding shell environment values."""
    if not path.is_file():
        return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, separator, value = line.partition("=")
            key = key.strip()
            if not separator or not key or key in os.environ:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ[key] = value
    except OSError:
        pass


_load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# --------------------------------------------------------------------------
# Tunable knobs. Kept at the top so they're easy to sweep during iteration.
# --------------------------------------------------------------------------
MAX_TURNS = 10
OVER_GENERALITY_THRESHOLD = 50   # candidate-pool size that triggers a clarifying ask
MIN_FILLED_SLOTS_BEFORE_ASK_STOPS = 3
CRITICAL_RATING_BIAS = 0.35      # quality nudge for "critical" raters
DEFAULT_RATING_BIAS = 0.12
SIZE_RELEVANT_HINTS = (
    "shirt", "dress", "jacket", "pant", "shoe", "boot", "bra", "jean",
    "skirt", "sweater", "coat", "legging", "short", "sock", "glove",
    "suit", "top", "hoodie", "swimsuit", "romper", "jumpsuit", "underwear",
)

# --------------------------------------------------------------------------
# Optional LLM assist. Fully inert unless a local HuggingFace model can be
# loaded -- the agent works entirely without it (no paid/hosted LLM
# required, no network call at all); when a local model IS available, it's
# used sparingly for the two things regex/bm25 are structurally bad at:
#   1. Slot extraction when the customer's phrasing doesn't hit our
#      keyword lists (e.g. "water resistant" when only "waterproof" is
#      in FEATURE_WORDS) -- called only when regex found nothing new this
#      turn, so it's a fallback, not a replacement.
#   2. Re-ranking a short bm25-retrieved candidate list using the full
#      conversation context, when bm25 alone can't discriminate between
#      many products sharing one generic term (e.g. "leather" matches
#      thousands of unrelated items; a model reading titles/features can
#      tell which one is actually a belt).
# Any missing dependency, model-load failure, generation timeout, or bad
# JSON falls straight back to the heuristic path with zero behavior
# change. There is no network dependency here (unlike the old Groq-backed
# version) -- everything runs from local weights.
# --------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # transformers/torch not installed -- LLM assist stays off
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

LLM_MODEL = os.environ.get("AGENT_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LLM_LOCAL_DIR = os.environ.get("AGENT_LLM_LOCAL_DIR", "")  # optional path to a pre-downloaded snapshot
LLM_DEVICE = os.environ.get(
    "AGENT_LLM_DEVICE",
    "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu",
)
LLM_DTYPE = os.environ.get("AGENT_LLM_DTYPE", "auto")
# Explicit opt-out even when the dependencies/model are available.
_LLM_DISABLED_BY_ENV = os.environ.get("AGENT_LLM_ENABLED", "1").lower() in {"0", "false", "no", "off"}
LLM_ENABLED = (AutoModelForCausalLM is not None) and not _LLM_DISABLED_BY_ENV
LLM_TIMEOUT_SECONDS = float(os.environ.get("AGENT_LLM_TIMEOUT_SECONDS", "15"))
LLM_RERANK_POOL_SIZE = 25   # candidates shown to the model for re-ranking
DEBUG = os.environ.get("AGENT_DEBUG", "").lower() in {"1", "true", "yes", "on"}
LOCAL_EMBEDDING_INDEX = os.environ.get("LOCAL_EMBEDDING_INDEX", str(DEFAULT_INDEX))
LOCAL_EMBEDDING_MODEL = os.environ.get(
    "LOCAL_EMBEDDING_MODEL",
    str(DEFAULT_MODEL_DIR if DEFAULT_MODEL_DIR.exists() else DEFAULT_MODEL),
)
LOCAL_EMBEDDING_CANDIDATES = 100

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "have", "has", "just", "still", "need", "needs", "like", "something",
}


def _debug(message: str) -> None:
    if DEBUG:
        print(f"[agent] {message}", flush=True)

# --------------------------------------------------------------------------
# Slot vocabularies (heuristic, regex-based -- no external NLP dependency,
# consistent with the in-memory / no-heavy-infra constraint).
# --------------------------------------------------------------------------
MATERIAL_WORDS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "denim", "linen", "suede", "canvas", "fleece", "cashmere",
    "down", "mesh", "fabric",
)
COLOR_WORDS = (
    "black", "white", "blue", "navy", "red", "pink", "green", "brown",
    "tan", "beige", "gray", "grey", "purple", "yellow", "orange",
    "maroon", "olive", "gold", "silver",
)
STYLE_WORDS = (
    "casual", "formal", "athletic", "vintage", "classic", "slim fit",
    "relaxed fit", "bohemian", "minimalist", "sporty", "elegant",
    "trendy", "preppy", "oversized",
)
USE_CASE_WORDS = (
    "running", "hiking", "gym", "workout", "office", "work", "wedding",
    "winter", "summer", "outdoor", "travel", "yoga", "party",
    "everyday", "school", "beach",
)
FEATURE_WORDS = (
    "waterproof", "breathable", "lightweight", "stretch", "stretchy",
    "adjustable", "quick-dry", "quick dry", "moisture-wicking",
    "moisture wicking", "non-slip", "insulated", "packable",
    "wrinkle-resistant", "machine washable",
)

SIZE_RE = re.compile(
    r"\b(size\s*\d{1,2}(?:\.\d)?|extra[- ]small|extra[- ]large|small|medium|"
    r"large|x{1,3}[- ]?large|xs|xl|xxl|xxxl)\b",
    re.IGNORECASE,
)
BUDGET_RE = re.compile(
    r"\$\s?(\d+(?:\.\d{1,2})?)|under\s+\$?(\d+)|budget(?:\s+of|\s+around)?\s*\$?(\d+)|"
    r"less than\s+\$?(\d+)|no more than\s+\$?(\d+)",
    re.IGNORECASE,
)
BRAND_RE = re.compile(
    r"\bbrand(?:\s+is|\s*[:\-])?\s+([A-Za-z][A-Za-z0-9&.'\- ]{1,24})|"
    r"\bfrom\s+([A-Z][A-Za-z0-9&.'\-]{1,24})\b"
)
CATEGORY_RE = re.compile(
    r"looking for ([a-zA-Z0-9 \-']{3,40}?)(?:[.,;]| but| that| with| in| for|$)",
    re.IGNORECASE,
)
OVERRIDE_RE = re.compile(
    r"\b(actually|ignore (?:that|my earlier|the earlier|previous)|"
    r"forget (?:that|it|what i said)|never mind|instead of that|"
    r"scratch that|change of plans|on second thought)\b",
    re.IGNORECASE,
)
BROWSING_HINT_RE = re.compile(
    r"\b(just looking|browsing|exploring|not sure|something like|"
    r"any suggestions|any recommendations|what (?:do you have|options)|"
    r"show me options|open to (?:anything|suggestions))\b",
    re.IGNORECASE,
)
BUYING_HINT_RE = re.compile(
    r"\b(need|must|required|exactly|i want to buy|purchase|has to be|"
    r"looking to buy|ready to buy|key requirement)\b",
    re.IGNORECASE,
)

# Attributes we treat as "hard" (used as required AND terms for the Buying
# track) versus "soft" (used only to bias/broaden ranking).
HARD_ATTRIBUTES = ("material", "color", "size", "brand", "budget")
SOFT_ATTRIBUTES = ("style", "use_case", "feature")

# The shopper profile is deliberately a fixed schema rather than an
# opportunistic bag of extracted terms.  `budget` is the public API name for
# the question, while `price_range` is the product-preference field it fills.
PRODUCT_FEATURE_FIELDS = (
    "category", "material", "color", "size", "price_range", "style", "feature",
)
NO_PREFERENCE_RE = re.compile(
    r"\b(?:no|don't have|do not have|without|doesn't matter|does not matter)\b.*\b(?:preference|preference for)\b|"
    r"\b(?:anything|any)\s+(?:is\s+)?fine\b",
    re.IGNORECASE,
)
QUESTION_TO_FEATURE_FIELD = {"budget": "price_range"}

ASK_PRIORITY_BUYING = ["material", "size", "color", "budget", "style", "brand", "use_case", "feature"]
ASK_PRIORITY_BROWSING = ["use_case", "category", "style", "material", "color", "budget", "brand", "feature"]
# These lists intentionally contain every field in PRODUCT_FEATURE_FIELDS.
# They are used while completing the fixed product-preference record; the
# broader lists above are retained for supplemental refinement afterwards.
CORE_ASK_PRIORITY_BUYING = ["category", "material", "size", "color", "budget", "style", "feature"]
CORE_ASK_PRIORITY_BROWSING = ["category", "style", "material", "color", "budget", "feature", "size"]

ASK_TEMPLATES = {
    "category": "What type of item are you shopping for?",
    "material": "Do you have a material preference, like cotton or leather?",
    "color": "Any color you're going for?",
    "size": "What size do you need?",
    "style": "What style or fit are you looking for?",
    "brand": "Is there a brand you prefer?",
    "budget": "Do you have a budget in mind?",
    "feature": "Any specific features that matter, like waterproof or lightweight?",
    "use_case": "What will you mainly use this for?",
    "other": "Is there anything else I should know to narrow this down?",
}

GENERIC_CATEGORY_WORDS = {"clothing", "item", "something", "product", "clothes"}


def _word_group_re(words: tuple[str, ...]) -> re.Pattern:
    escaped = sorted((re.escape(w) for w in words), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


MATERIAL_RE = _word_group_re(MATERIAL_WORDS)
COLOR_RE = _word_group_re(COLOR_WORDS)
STYLE_RE = _word_group_re(STYLE_WORDS)
USE_CASE_RE = _word_group_re(USE_CASE_WORDS)
FEATURE_RE = _word_group_re(FEATURE_WORDS)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _match_term(term: str) -> str:
    """Build an FTS5 match fragment for a slot value.

    Multi-word values (e.g. "slim fit", or a whole category phrase like
    "Bras Everyday Bras") are expanded into an OR-group of their individual
    words rather than quoted as one exact adjacent phrase. Quoting the
    whole phrase requires those exact words to appear adjacent and in that
    order in the catalog text, which almost never happens and silently
    makes the term useless -- verified during debugging that a category
    phrase quoted whole matched effectively nothing.
    """
    words = [w for w in TOKEN_RE.findall(term) if w]
    if not words:
        return ""
    if len(words) == 1:
        return f'"{words[0]}"'
    return "(" + " OR ".join(f'"{w}"' for w in words) + ")"


def extract_slots(message: str) -> tuple[dict[str, str], float | None]:
    """Pull hard/soft attribute values out of a raw customer message.

    Returns (slots, budget_value). `slots` maps attribute name -> the raw
    matched phrase (used as an FTS search term); `budget_value` is a parsed
    numeric ceiling used as a SQL price filter, kept separate from the text
    slot because "$50" isn't a useful full-text search token.
    """
    slots: dict[str, str] = {}
    budget_value: float | None = None

    if m := MATERIAL_RE.search(message):
        slots["material"] = m.group(1).lower()
    if m := COLOR_RE.search(message):
        slots["color"] = m.group(1).lower()
    if m := SIZE_RE.search(message):
        slots["size"] = m.group(1).lower()
    if m := STYLE_RE.search(message):
        slots["style"] = m.group(1).lower()
    if m := USE_CASE_RE.search(message):
        slots["use_case"] = m.group(1).lower()
    if m := FEATURE_RE.search(message):
        slots["feature"] = m.group(1).lower()
    if m := BRAND_RE.search(message):
        slots["brand"] = (m.group(1) or m.group(2) or "").strip()
    if m := BUDGET_RE.search(message):
        raw = next((g for g in m.groups() if g), None)
        if raw:
            try:
                budget_value = float(raw)
                slots["budget"] = f"under ${int(budget_value)}"
            except ValueError:
                pass
    if m := CATEGORY_RE.search(message):
        category = m.group(1).strip()
        if category:
            slots["category"] = category

    return slots, budget_value


# --------------------------------------------------------------------------
# Local HuggingFace model loading. Lazy + guarded by a lock so concurrent
# sessions don't race to load the model twice; loaded once per process and
# reused for every _call_llm invocation thereafter.
# --------------------------------------------------------------------------
_llm_lock = threading.Lock()
_llm_tokenizer = None
_llm_model = None
_llm_load_failed = False
_llm_executor: ThreadPoolExecutor | None = None


def _resolve_dtype():
    if torch is None:
        return None
    if LLM_DTYPE == "auto":
        return "auto"
    return getattr(torch, LLM_DTYPE, "auto")


def _get_llm():
    """Load (once) and return (tokenizer, model), or None if unavailable.

    Any exception during load permanently disables the LLM path for the
    rest of the process (mirrors the old "bad key / network down" fallback
    behavior of the Groq client, just for local-load failures instead).
    """
    global _llm_tokenizer, _llm_model, _llm_load_failed, LLM_ENABLED

    if not LLM_ENABLED or _llm_load_failed:
        return None
    if _llm_model is not None:
        return _llm_tokenizer, _llm_model

    with _llm_lock:
        if _llm_model is not None:
            return _llm_tokenizer, _llm_model
        if _llm_load_failed:
            return None
        model_path = LLM_LOCAL_DIR or LLM_MODEL
        try:
            _debug(f"loading local HF model '{model_path}' on device={LLM_DEVICE}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=_resolve_dtype(),
            )
            model.to(LLM_DEVICE)
            model.eval()
            _llm_tokenizer, _llm_model = tokenizer, model
            _debug("local HF model loaded successfully")
        except Exception as error:
            _debug(f"failed to load local HF model ({error}); LLM assist disabled for this run")
            _llm_load_failed = True
            LLM_ENABLED = False
            return None
    return _llm_tokenizer, _llm_model


def _get_executor() -> ThreadPoolExecutor:
    global _llm_executor
    if _llm_executor is None:
        with _llm_lock:
            if _llm_executor is None:
                _llm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-generate")
    return _llm_executor


def _generate(system: str, user: str, max_tokens: int) -> tuple[str, dict]:
    """Runs on the worker thread; raises on any failure so the caller's
    timeout/except wrapper can fall back to heuristics uniformly.
    """
    tokenizer, model = _get_llm()  # already loaded by caller, cheap re-check
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"{system}\n\n{user}\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    usage = {
        "input_tokens": int(input_len),
        "output_tokens": int(generated_ids.shape[0]),
    }
    return text, usage


def _call_llm(system: str, user: str, max_tokens: int) -> tuple[str, dict] | None:
    """Single-turn call to a local HuggingFace causal LM. Returns (text, usage)
    or None on any failure (no dependencies, load failure, generation
    timeout, or unexpected error). Generation runs on a bounded worker
    thread so a slow/hung local model degrades to the heuristic path
    instead of blocking the request indefinitely, matching the network
    -timeout behavior of the API-backed version this replaces.
    """
    if _get_llm() is None:
        return None
    try:
        _debug(f"generating locally with model={LLM_LOCAL_DIR or LLM_MODEL} max_tokens={max_tokens}")
        future = _get_executor().submit(_generate, system, user, max_tokens)
        text, usage = future.result(timeout=LLM_TIMEOUT_SECONDS)
        _debug(
            "local generation finished "
            f"input_tokens={usage['input_tokens']} output_tokens={usage['output_tokens']}"
        )
        return text, usage
    except FutureTimeoutError:
        _debug(f"local generation exceeded {LLM_TIMEOUT_SECONDS}s timeout; using heuristic fallback")
        return None
    except Exception as error:
        _debug(f"local generation failed ({error}); using heuristic fallback")
        return None


def _parse_json_block(text: str) -> object | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned.strip())
    except (json.JSONDecodeError, ValueError):
        return None


def llm_extract_slots(message: str, missing_attributes: list[str]) -> tuple[dict[str, str], dict]:
    """Fallback slot extraction for phrasing the regex vocabulary misses
    (e.g. "water resistant" when only "waterproof" is a known keyword).
    Only worth calling when the cheap regex pass found nothing new.
    """
    empty_usage = {"input_tokens": 0, "output_tokens": 0}
    if not missing_attributes:
        return {}, empty_usage
    system = (
        "You extract shopping preference attributes from one customer message. "
        "Reply with strict minified JSON only -- no prose, no markdown fences. "
        "Only include keys from the allowed list where the message states a clear value. "
        "Values must be short, at most 4 words."
    )
    user = (
        f"Allowed attribute keys: {', '.join(missing_attributes)}.\n"
        f"Customer message: {message!r}\n"
        'Example reply shape: {"material": "leather", "feature": "water resistant"}'
    )
    result = _call_llm(system, user, max_tokens=150)
    if result is None:
        return {}, empty_usage
    text, usage = result
    data = _parse_json_block(text)
    if not isinstance(data, dict):
        return {}, usage
    extracted = {
        key: str(value)[:60]
        for key, value in data.items()
        if key in missing_attributes and value
    }
    return extracted, usage


def llm_rerank(context_summary: str, candidates: list[tuple[str, str]]) -> tuple[list[str] | None, dict]:
    """Re-order a short bm25 candidate list using full conversation context.

    bm25 can't tell apart many products that share one generic matched
    term (e.g. "leather" hits everything from belts to jackets to boots);
    a model reading the actual titles against what the customer asked for
    can. Returns None (keep heuristic order) on any failure so this is
    purely additive.
    """
    empty_usage = {"input_tokens": 0, "output_tokens": 0}
    if not candidates:
        return None, empty_usage
    listing = "\n".join(f"{i + 1}. [{asin}] {text[:160]}" for i, (asin, text) in enumerate(candidates))
    system = (
        "You rank e-commerce product candidates by fit to a customer's stated needs. "
        "Reply with strict minified JSON only: an array of parent_asin strings, "
        "best match first, using only ids from the given list. No other text."
    )
    user = f"Customer needs so far: {context_summary}\n\nCandidates:\n{listing}\n\nReturn the ranked array."
    result = _call_llm(system, user, max_tokens=400)
    if result is None:
        return None, empty_usage
    text, usage = result
    order = _parse_json_block(text)
    if not isinstance(order, list):
        return None, usage
    valid_ids = [asin for asin, _ in candidates]
    valid_set = set(valid_ids)
    ranked = [item for item in order if isinstance(item, str) and item in valid_set]
    for asin in valid_ids:
        if asin not in ranked:
            ranked.append(asin)  # keep any model-dropped candidates as a fallback tail
    return ranked, usage


@dataclass
class SessionState:
    profile: dict = field(default_factory=dict)
    # Always contains all seven core product-preference fields.  A value of
    # None means that the customer has not supplied that preference yet.
    product_features: dict[str, str | None] = field(
        default_factory=lambda: dict.fromkeys(PRODUCT_FEATURE_FIELDS)
    )
    slots: dict[str, str] = field(default_factory=dict)
    budget_value: float | None = None
    hard_order: list[str] = field(default_factory=list)   # order slots were filled, for relaxation
    asked_attributes: set[str] = field(default_factory=set)
    last_asked_attribute: str | None = None
    intent: str | None = None
    turns_seen: int = 0

    def filled_relevant_count(self) -> int:
        return len([a for a in HARD_ATTRIBUTES + SOFT_ATTRIBUTES if a in self.slots])

    def set_slot(self, attribute: str, value: str) -> None:
        """Persist a raw search slot and mirror it into the fixed schema."""
        self.slots[attribute] = value
        feature_field = QUESTION_TO_FEATURE_FIELD.get(attribute, attribute)
        if feature_field in self.product_features:
            self.product_features[feature_field] = value


class Agent:
    """Intent-routed, stateful shopping agent with adaptive clarification.

    Architecture:
      I.   Intent routing -> buying (hard-filtered) vs browsing (broad OR) track.
      II.  Dialog state machine -> accumulates slots across turns; detects
           override language and wipes prior constraints.
      III. Adaptive orchestration -> if a strict query returns nothing, it
           relaxes constraints one at a time (self-correcting retrieval)
           instead of failing; if a query is over-broad, it asks instead of
           dumping generic results.
      IV.  Personalization -> user_profile.rating_style nudges ranking
           toward higher-rated products for "critical" raters.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._states: dict[str, SessionState] = {}
        _debug(f"building catalog index from {self.catalog_path}")
        self._build_index()
        self.embedding_index = self._load_embedding_index()
        _debug(f"ready; dense retrieval={'enabled' if self.embedding_index else 'disabled'}")
        if LLM_ENABLED:
            # Warm the local model at construction time rather than on the
            # first request, so latency shows up in startup, not turn 1.
            _get_llm()

    @staticmethod
    def _load_embedding_index():
        index_path = Path(LOCAL_EMBEDDING_INDEX)
        if LocalEmbeddingIndex is None:
            _debug("embedding support is unavailable; skipping embedding index load")
            return None
        if not index_path.exists():
            _debug(f"embedding index not found at {index_path}; skipping embedding build and using BM25")
            return None
        try:
            _debug(f"found embedding index at {index_path}; skipping embedding build and loading it")
            return LocalEmbeddingIndex(
                index_path,
                LOCAL_EMBEDDING_MODEL,
                expected_model=DEFAULT_MODEL,
            )
        except Exception as error:
            _debug(f"could not load existing embedding index ({error}); using BM25")
            return None

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "price UNINDEXED, average_rating UNINDEXED, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                        "" if product.get("price") in (None, "") else str(product["price"]),
                        "" if product.get("average_rating") in (None, "") else str(product["average_rating"]),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany(
                        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch
                    )
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    # ---------------------------------------------------------------- reset
    def reset(self, session_id: str, user_profile: dict) -> None:
        self._states[session_id] = SessionState(profile=user_profile or {})

    # -------------------------------------------------------------- respond
    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._states.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        state.turns_seen = turn
        _debug(f"session={session_id} turn={turn} question={user_message!r}")

        # A customer saying that a requested preference does not matter is a
        # useful answer, not a missing value.  Record it in the fixed schema
        # without putting a meaningless "any" term into catalog retrieval.
        if state.last_asked_attribute and NO_PREFERENCE_RE.search(user_message):
            feature_field = QUESTION_TO_FEATURE_FIELD.get(state.last_asked_attribute, state.last_asked_attribute)
            if feature_field in state.product_features:
                state.product_features[feature_field] = "any"

        # II. Dialog state machine: override detection wipes prior *hard*
        # constraints only. Category and soft attributes (style/use_case/
        # feature) are kept, since overrides in practice replace one
        # specific preference rather than the whole conversation. Clearing
        # the attribute from asked_attributes lets the agent re-ask it if
        # it turns out to still matter.
        if state.slots and OVERRIDE_RE.search(user_message):
            # Preserve the non-hard descriptive context, but reset the fixed
            # feature record so it accurately represents the active request.
            kept_slots = {
                attribute: state.slots[attribute]
                for attribute in ("category", *SOFT_ATTRIBUTES)
                if attribute in state.slots
            }
            state.slots = {}
            state.product_features = dict.fromkeys(PRODUCT_FEATURE_FIELDS)
            for attribute, value in kept_slots.items():
                state.set_slot(attribute, value)
            state.hard_order = []
            state.budget_value = None

        new_slots, new_budget = extract_slots(user_message)
        gained_hard_slot = False
        for attribute, value in new_slots.items():
            is_new_or_changed = state.slots.get(attribute) != value
            state.set_slot(attribute, value)
            if attribute in HARD_ATTRIBUTES and is_new_or_changed:
                if attribute in state.hard_order:
                    state.hard_order.remove(attribute)
                state.hard_order.append(attribute)
                gained_hard_slot = True
        if new_budget is not None:
            state.budget_value = new_budget
            if "budget" not in state.hard_order:
                state.hard_order.append("budget")
            gained_hard_slot = True

        llm_input_tokens = 0
        llm_output_tokens = 0

        # LLM slot-extraction fallback: only worth the call when the cheap
        # regex pass found nothing new this turn but attributes remain
        # unfilled -- catches phrasing regex vocab lists miss entirely
        # (e.g. "water resistant" vs. the known keyword "waterproof").
        if LLM_ENABLED and not new_slots and new_budget is None:
            missing = [a for a in HARD_ATTRIBUTES + SOFT_ATTRIBUTES if a not in state.slots]
            llm_slots, usage = llm_extract_slots(user_message, missing)
            llm_input_tokens += usage["input_tokens"]
            llm_output_tokens += usage["output_tokens"]
            for attribute, value in llm_slots.items():
                state.set_slot(attribute, value)
                if attribute in HARD_ATTRIBUTES:
                    if attribute in state.hard_order:
                        state.hard_order.remove(attribute)
                    state.hard_order.append(attribute)
                    gained_hard_slot = True

        # I. Intent routing (sticky toward "buying" once a hard constraint appears).
        if gained_hard_slot or BUYING_HINT_RE.search(user_message):
            state.intent = "buying"
        elif state.intent is None:
            state.intent = "browsing" if BROWSING_HINT_RE.search(user_message) else "buying"

        # Build search term pools.
        required_terms = [state.slots[a] for a in state.hard_order if a in state.slots]
        optional_terms: list[str] = []
        for attribute in SOFT_ATTRIBUTES + ("category",):
            if attribute in state.slots:
                optional_terms.append(state.slots[attribute])
        optional_terms.extend(_terms(user_message))
        optional_terms.extend(state.profile.get("preference_tags") or [])
        optional_terms = list(dict.fromkeys(t for t in optional_terms if t))

        if state.intent == "browsing":
            # Browsing track: don't hard-filter on constraints yet, treat
            # everything as a broad OR so results stay diverse.
            active_required = required_terms if len(required_terms) >= 2 else []
            active_optional = optional_terms + [t for t in required_terms if t not in active_required]
        else:
            active_required = list(required_terms)
            active_optional = optional_terms

        rating_bias = (
            CRITICAL_RATING_BIAS if "critical" in str(state.profile.get("rating_style", "")).lower()
            else DEFAULT_RATING_BIAS
        )

        retrieval_k = (
            max(top_k, LOCAL_EMBEDDING_CANDIDATES)
            if self.embedding_index is not None
            else top_k
        )
        recommendations, candidate_count, expression, budget_used = self._search(
            active_required, active_optional, state.budget_value, rating_bias, retrieval_k
        )

        # Dense retrieval is a recall-oriented second route.  It is merged
        # with FTS results before the deterministic feature-aware reranker.
        if self.embedding_index is not None:
            dense_query = self._describe_state(state) + "\nUser: " + user_message
            dense_candidates = self.embedding_index.search(
                dense_query, LOCAL_EMBEDDING_CANDIDATES
            )
            recommendations = self._hybrid_rank(
                recommendations, dense_candidates, active_required, optional_terms,
                state, rating_bias, top_k
            )

        # LLM semantic re-ranking: bm25 can't discriminate between many
        # products sharing one generic matched term (e.g. "leather" hits
        # thousands of unrelated items). Only worth the call on the Buying
        # track with real constraints in play, where ranking precision
        # (MRR) is what's being left on the table.
        if LLM_ENABLED and state.intent == "buying" and expression:
            pool = self._fetch_titles(expression, budget_used, rating_bias, LLM_RERANK_POOL_SIZE)
            if pool:
                context_summary = self._describe_state(state)
                reranked, usage = llm_rerank(context_summary, pool)
                llm_input_tokens += usage["input_tokens"]
                llm_output_tokens += usage["output_tokens"]
                if reranked:
                    recommendations = reranked[:top_k]

        over_generality = candidate_count > OVER_GENERALITY_THRESHOLD
        ask_attribute = self._choose_ask_attribute(state, over_generality, turn)
        _debug(
            f"session={session_id} candidates={candidate_count} "
            f"recommendations={len(recommendations)} next_question={ask_attribute or 'none'}"
        )
        if ask_attribute:
            state.asked_attributes.add(ask_attribute)
            state.last_asked_attribute = ask_attribute
            message = ASK_TEMPLATES[ask_attribute]
        elif recommendations:
            state.last_asked_attribute = None
            message = f"Here are {len(recommendations)} matches based on what you've told me so far."
        else:
            state.last_asked_attribute = None
            message = "I couldn't find a close match yet -- could you tell me more about what you need?"

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [{"parent_asin": asin} for asin in recommendations],
            "usage": {
                "prompt_tokens": len(user_message.split()) + llm_input_tokens,
                "completion_tokens": len(message.split()) + llm_output_tokens,
            },
        }

    @staticmethod
    def _describe_state(state: SessionState) -> str:
        parts = [f"{key}: {value}" for key, value in state.slots.items()]
        tags = state.profile.get("preference_tags") or []
        if tags:
            parts.append("general preferences: " + ", ".join(tags))
        return "; ".join(parts) if parts else "no specific constraints stated yet"

    def _hybrid_rank(
        self,
        lexical: list[str],
        dense: list[tuple[str, float]],
        required_terms: list[str],
        optional_terms: list[str],
        state: SessionState,
        rating_bias: float,
        top_k: int,
    ) -> list[str]:
        """Fuse FTS and dense candidates with cheap catalog-field boosts.

        Dense retrieval supplies recall; lexical retrieval and explicit slots
        supply precision.  Candidates absent from both routes are never
        introduced, preserving the catalog-only output contract.
        """
        dense_rank = {asin: rank for rank, (asin, _) in enumerate(dense)}
        pool = list(dict.fromkeys(lexical + [asin for asin, _ in dense]))
        if not pool:
            return []
        lexical_rank = {asin: rank for rank, asin in enumerate(lexical)}
        scored: list[tuple[float, str]] = []
        for asin in pool:
            # Raw cosine and BM25 scores are not calibrated to each other.
            # Reciprocal rank fusion keeps either route from overwhelming the
            # other because of an arbitrary score scale.
            lexical_score = 1.0 / (60.0 + lexical_rank.get(asin, 10_000))
            dense_score = 1.0 / (60.0 + dense_rank.get(asin, 10_000))
            score = 0.65 * lexical_score + 0.35 * dense_score
            if state.profile.get("rating_style") == "critical":
                score += 0.005 * rating_bias
            scored.append((score, asin))
        scored.sort(reverse=True)
        return [asin for _, asin in scored[:top_k]]


    # -------------------------------------------------------- ask selection
    def _choose_ask_attribute(self, state: SessionState, over_generality: bool, turn: int) -> str | None:
        if turn >= MAX_TURNS:
            return None  # no more customer turns will follow; asking wastes it

        # Completion of the seven-field preference record is the primary
        # dialogue goal.  We keep asking for the next missing field even when
        # retrieval is already narrow, so later turns improve both recall and
        # the explanation of why a product was recommended.  `budget` is the
        # API-facing question name for the `price_range` state field.
        missing_core = [
            feature for feature in PRODUCT_FEATURE_FIELDS
            if state.product_features[feature] is None
        ]
        if missing_core:
            core_priority = (
                CORE_ASK_PRIORITY_BUYING
                if state.intent == "buying"
                else CORE_ASK_PRIORITY_BROWSING
            )
            for attribute in core_priority:
                feature_field = QUESTION_TO_FEATURE_FIELD.get(attribute, attribute)
                if feature_field not in missing_core or attribute in state.asked_attributes:
                    continue
                return attribute

        # Once the core record is complete, only ask a further question when
        # the current retrieval is still too broad.  These can be the
        # supplemental brand/use_case fields, which remain useful for search
        # but are not part of the seven required product features.
        if not over_generality:
            return None
        priority = ASK_PRIORITY_BUYING if state.intent == "buying" else ASK_PRIORITY_BROWSING
        category_text = state.slots.get("category", "").lower()
        for attribute in priority:
            if attribute in state.asked_attributes:
                continue
            if attribute == "category":
                if category_text and category_text not in GENERIC_CATEGORY_WORDS:
                    continue
            elif attribute == "size":
                # Don't waste a turn asking about size for non-wearable
                # products (jewelry, accessories, home goods, etc.).
                if category_text and not any(hint in category_text for hint in SIZE_RELEVANT_HINTS):
                    continue
            elif attribute in state.slots:
                continue
            return attribute
        return None

    # --------------------------------------------------------------- search
    def _search(
        self,
        required_terms: list[str],
        optional_terms: list[str],
        budget_value: float | None,
        rating_bias: float,
        top_k: int,
    ) -> tuple[list[str], int, str, float | None]:
        """Retrieve with a precision-to-recall fallback cascade.

        Each tier is a (required, optional) pairing tried in decreasing
        order of precision. A tier "succeeds" once it has at least one
        match; we don't require it to contain the true target (we can't
        know that), just that the query isn't degenerate. This is the
        "runtime re-orchestration" behavior: instead of committing to one
        fixed query shape, the agent adaptively broadens when a stricter
        combination comes back empty.

        Design note: soft (optional) terms are combined with required terms
        as `required AND (opt1 OR opt2 OR ...)` when both are present --
        this keeps results topically on-target (e.g. an accessory query
        shouldn't surface bags and totes just because they share a material
        word). But that combination is only the *first* tier tried, not the
        only one: if it returns nothing, we fall back to required-only,
        then to progressively relaxed required terms, then to optional-only,
        so a real match is never lost just because one turn's phrasing
        didn't overlap with the catalog text.
        """
        price_filter_active = budget_value is not None
        working_required = list(required_terms)

        def budget_for(active: bool) -> float | None:
            return budget_value if active else None

        # Tier 1: full precision (required AND optional), full budget.
        tiers: list[tuple[list[str], list[str]]] = []
        if working_required and optional_terms:
            tiers.append((working_required, optional_terms))
        if working_required:
            tiers.append((working_required, []))

        chosen_required, chosen_optional = working_required, optional_terms
        expression = ""
        count = 0
        for req, opt in tiers:
            expression = self._build_expression(req, opt)
            count = self._count_matches(expression, budget_for(price_filter_active))
            if count > 0:
                chosen_required, chosen_optional = req, opt
                break
        else:
            # Progressive relaxation: drop the oldest hard constraint first,
            # then drop the budget cap, then fall back to optional-only,
            # then to no filter at all.
            relaxed = list(working_required)
            found = False
            while relaxed:
                relaxed = relaxed[1:]
                expression = self._build_expression(relaxed, [])
                count = self._count_matches(expression, budget_for(price_filter_active))
                if count > 0:
                    chosen_required, chosen_optional = relaxed, []
                    found = True
                    break
            if not found and price_filter_active:
                price_filter_active = False
                expression = self._build_expression([], optional_terms)
                count = self._count_matches(expression, None)
                if count > 0:
                    chosen_required, chosen_optional = [], optional_terms
                    found = True
            if not found and optional_terms:
                expression = self._build_expression([], optional_terms)
                count = self._count_matches(expression, budget_for(price_filter_active))
                if count > 0:
                    chosen_required, chosen_optional = [], optional_terms
                    found = True
            if not found:
                chosen_required, chosen_optional = [], []
                expression = ""
                count = 0

        recommendations = self._fetch(
            expression, budget_for(price_filter_active), rating_bias, top_k
        )
        return recommendations, count, expression, budget_for(price_filter_active)

    @staticmethod
    def _build_expression(required_terms: list[str], optional_terms: list[str]) -> str:
        required = " AND ".join(_match_term(t) for t in required_terms if _match_term(t))
        optional = " OR ".join(_match_term(t) for t in dict.fromkeys(optional_terms) if _match_term(t))
        if required and optional:
            return f"({required}) AND ({optional})"
        return required or optional

    def _count_matches(self, expression: str, budget_value: float | None) -> int:
        if not expression:
            return 0
        sql = "SELECT COUNT(*) FROM products WHERE products MATCH ?"
        params: list = [expression]
        if budget_value is not None:
            sql += " AND (price = '' OR CAST(price AS REAL) <= ?)"
            params.append(budget_value)
        try:
            return int(self.connection.execute(sql, params).fetchone()[0])
        except sqlite3.OperationalError:
            return 0

    def _fetch(
        self, expression: str, budget_value: float | None, rating_bias: float, top_k: int
    ) -> list[str]:
        params: list = []
        if expression:
            sql = "SELECT parent_asin FROM products WHERE products MATCH ?"
            params.append(expression)
            if budget_value is not None:
                sql += " AND (price = '' OR CAST(price AS REAL) <= ?)"
                params.append(budget_value)
            sql += (
                " ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0, 0.0, 0.0)"
                f" - {rating_bias} * COALESCE(NULLIF(average_rating, ''), 0) LIMIT ?"
            )
            params.append(top_k)
        else:
            sql = (
                "SELECT parent_asin FROM products "
                "ORDER BY CAST(NULLIF(average_rating, '') AS REAL) DESC LIMIT ?"
            )
            params.append(top_k)
        try:
            rows = self.connection.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        return [str(row[0]) for row in rows]

    def _fetch_titles(
        self, expression: str, budget_value: float | None, rating_bias: float, pool_size: int
    ) -> list[tuple[str, str]]:
        """Same ordering as _fetch, but returns (asin, title+features text)
        for the LLM re-ranking pass instead of just the asin.
        """
        if not expression:
            return []
        params: list = [expression]
        sql = (
            "SELECT parent_asin, title || ' -- ' || features AS blurb "
            "FROM products WHERE products MATCH ?"
        )
        if budget_value is not None:
            sql += " AND (price = '' OR CAST(price AS REAL) <= ?)"
            params.append(budget_value)
        sql += (
            " ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0, 0.0, 0.0)"
            f" - {rating_bias} * COALESCE(NULLIF(average_rating, ''), 0) LIMIT ?"
        )
        params.append(pool_size)
        try:
            rows = self.connection.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(str(row[0]), str(row[1])) for row in rows]