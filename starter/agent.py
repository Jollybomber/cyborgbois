from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from local_embeddings import DEFAULT_INDEX, DEFAULT_MODEL, DEFAULT_MODEL_DIR, LocalEmbeddingIndex
except Exception:  # Dense retrieval is optional and must never break lexical search.
    DEFAULT_INDEX = Path("data/catalog_bge_embeddings.npz")
    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    DEFAULT_MODEL_DIR = Path("data/bge-small-en-v1.5")
    LocalEmbeddingIndex = None


def _load_dotenv(path: Path) -> None:
    """Load a small .env file without overriding real environment values."""
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

MAX_TURNS = 10
MAX_CLARIFICATION_QUESTIONS = int(os.environ.get("AGENT_MAX_QUESTIONS", "6"))
OVER_GENERALITY_THRESHOLD = int(os.environ.get("AGENT_BROAD_POOL", "50"))
LEXICAL_CANDIDATES = int(os.environ.get("AGENT_LEXICAL_CANDIDATES", "200"))
DENSE_CANDIDATES = int(os.environ.get("AGENT_DENSE_CANDIDATES", "200"))
LLM_RERANK_POOL_SIZE = int(os.environ.get("AGENT_LLM_RERANK_POOL", "30"))
LLM_TIMEOUT_SECONDS = float(os.environ.get("AGENT_LLM_TIMEOUT", "8"))
DEBUG = os.environ.get("AGENT_DEBUG", "").lower() in {"1", "true", "yes", "on"}
LLM_RERANK_ENABLED = os.environ.get("AGENT_LLM_RERANK", "1").lower() not in {
    "0", "false", "no", "off"
}

LOCAL_EMBEDDING_INDEX = os.environ.get("LOCAL_EMBEDDING_INDEX", str(DEFAULT_INDEX))
LOCAL_EMBEDDING_MODEL = os.environ.get(
    "LOCAL_EMBEDDING_MODEL",
    str(DEFAULT_MODEL_DIR if DEFAULT_MODEL_DIR.exists() else DEFAULT_MODEL),
)

# Prefer Groq when both are configured; Anthropic remains backwards compatible.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if GROQ_API_KEY:
    LLM_PROVIDER = "groq"
    LLM_API_KEY = GROQ_API_KEY
    LLM_MODEL = os.environ.get("AGENT_LLM_MODEL", "openai/gpt-oss-20b")
elif ANTHROPIC_API_KEY:
    LLM_PROVIDER = "anthropic"
    LLM_API_KEY = ANTHROPIC_API_KEY
    LLM_MODEL = os.environ.get("AGENT_LLM_MODEL", "claude-haiku-4-5-20251001")
else:
    LLM_PROVIDER = "none"
    LLM_API_KEY = ""
    LLM_MODEL = os.environ.get("AGENT_LLM_MODEL", "")
LLM_ENABLED = bool(LLM_API_KEY)


def _debug(message: str) -> None:
    if DEBUG:
        print(f"[agent] {message}", flush=True)


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "have", "has", "just", "still", "need", "needs", "like", "something",
    "those", "what", "yet", "about", "tell", "could", "customer", "product",
}

MATERIAL_WORDS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "denim", "linen", "suede", "canvas", "fleece", "cashmere",
    "down", "mesh", "fabric", "stainless steel",
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
    "winter", "summer", "outdoor", "travel", "yoga", "party", "everyday",
    "school", "beach", "walking",
)
FEATURE_WORDS = (
    "waterproof", "water resistant", "breathable", "lightweight", "stretch",
    "stretchy", "adjustable", "quick-dry", "quick dry", "moisture-wicking",
    "moisture wicking", "non-slip", "insulated", "packable",
    "wrinkle-resistant", "machine washable", "buckle closure", "zipper closure",
)


def _word_group_re(words: tuple[str, ...]) -> re.Pattern[str]:
    escaped = sorted((re.escape(word) for word in words), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


MATERIAL_RE = _word_group_re(MATERIAL_WORDS)
COLOR_RE = _word_group_re(COLOR_WORDS)
STYLE_RE = _word_group_re(STYLE_WORDS)
USE_CASE_RE = _word_group_re(USE_CASE_WORDS)
FEATURE_RE = _word_group_re(FEATURE_WORDS)
SIZE_RE = re.compile(
    r"\b(size\s*\d{1,2}(?:\.\d)?|extra[- ]small|extra[- ]large|small|medium|"
    r"large|x{1,3}[- ]?large|xs|xl|xxl|xxxl|\d{1,2}(?:\.\d)?\s*(?:wide|narrow))\b",
    re.IGNORECASE,
)
BUDGET_RE = re.compile(
    r"\$\s?(\d+(?:\.\d{1,2})?)|under\s+\$?(\d+)|"
    r"budget(?:\s+of|\s+around)?\s*\$?(\d+)|less than\s+\$?(\d+)|"
    r"no more than\s+\$?(\d+)",
    re.IGNORECASE,
)
BRAND_RE = re.compile(
    r"\bbrand(?:\s+is|\s*[:\-])?\s+([A-Za-z][A-Za-z0-9&.'\- ]{1,24})|"
    r"\bfrom\s+([A-Z][A-Za-z0-9&.'\-]{1,24})\b"
)
CATEGORY_RE = re.compile(
    r"looking for ([a-zA-Z0-9 &\-']{3,80}?)(?:[.,;]|\s+but\b|\s+that\b|\s+with\b|$)",
    re.IGNORECASE,
)
OVERRIDE_RE = re.compile(
    r"\b(actually|ignore (?:that|my earlier|the earlier|previous)|"
    r"forget (?:that|it|what i said)|never mind|instead of that|scratch that|"
    r"change of plans|on second thought)\b",
    re.IGNORECASE,
)
BROWSING_HINT_RE = re.compile(
    r"\b(just looking|browsing|exploring|not sure|something like|any suggestions|"
    r"any recommendations|what (?:do you have|options)|show me options|"
    r"open to (?:anything|suggestions))\b",
    re.IGNORECASE,
)
BUYING_HINT_RE = re.compile(
    r"\b(need|must|required|exactly|i want to buy|purchase|has to be|"
    r"looking to buy|ready to buy|key requirement)\b",
    re.IGNORECASE,
)
NO_PREFERENCE_RE = re.compile(
    r"\b(?:no|don't have|do not have|without|doesn't matter|does not matter)\b.*"
    r"\b(?:preference|preference for)\b|\b(?:anything|any)\s+(?:is\s+)?fine\b|"
    r"\bno additional preference\b",
    re.IGNORECASE,
)
ANSWER_VALUE_RE = re.compile(
    r"(?:what matters is|key requirement is|what i need is|prioritize)\s*:\s*(.+)$",
    re.IGNORECASE,
)
EXCLUSION_RE = re.compile(
    r"\b(?:avoid|without|except|must not be|should not be)\s+"
    r"([a-z0-9][a-z0-9 &'\-]{1,60}?)(?:[.,;]|$)",
    re.IGNORECASE,
)

HARD_ATTRIBUTES = ("material", "color", "size", "brand", "budget")
ALLOWED_ASK_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand", "budget",
    "feature", "use_case", "other",
}
PRODUCT_FEATURE_FIELDS = (
    "category", "material", "color", "size", "price_range", "style", "feature",
)
QUESTION_TO_FEATURE_FIELD = {"budget": "price_range"}
SINGLE_VALUE_ATTRIBUTES = {
    "category", "material", "color", "size", "brand", "budget", "style", "use_case",
}
# These are product-domain priors, not evaluator-label frequencies. They only
# break close information-gain ties; the live candidate distribution selects
# the actual question on every turn.
QUESTION_UTILITY = {
    "feature": 1.15,
    "material": 1.0,
    "use_case": 0.95,
    "style": 0.9,
    "color": 0.7,
    "budget": 0.5,
    "size": 0.55,
    "brand": 0.15,
}
QUESTION_FALLBACK = [
    "use_case", "material", "feature", "style", "color", "budget", "size", "brand", "other"
]
ASK_TEMPLATES = {
    "category": "What type of item are you shopping for?",
    "material": "Do you have a material preference, like cotton or leather?",
    "color": "Any color you're going for?",
    "size": "What size do you need?",
    "style": "What style or fit are you looking for?",
    "brand": "Is there a brand you prefer?",
    "budget": "Do you have a budget in mind?",
    "feature": "Which product features or construction details matter most?",
    "use_case": "What will you mainly use this for?",
    "other": "Is there one other requirement that would narrow this down?",
}
SIZE_RELEVANT_HINTS = (
    "shirt", "dress", "jacket", "pant", "shoe", "boot", "bra", "jean", "skirt",
    "sweater", "coat", "legging", "short", "sock", "glove", "suit", "top",
    "hoodie", "swimsuit", "romper", "jumpsuit", "underwear",
)
CLOSURE_RE = re.compile(
    r"\b(pull on|button|zipper|buckle|hook and eye|lace up|slip on|snap|drawstring)\s+closure\b",
    re.IGNORECASE,
)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(_text(item) for item in value)
    return str(value)


def _clean_value(value: object, limit: int = 240) -> str:
    return re.sub(r"\s+", " ", str(value)).strip(" \t\n-;,.")[:limit].rstrip()


def _terms(text: str) -> list[str]:
    return [
        token.lower() for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _unique_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = _clean_value(value)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _match_term(term: str) -> str:
    words = list(dict.fromkeys(_terms(term)))
    if not words:
        return ""
    if len(words) == 1:
        return f'"{words[0]}"'
    return "(" + " OR ".join(f'"{word}"' for word in words) + ")"


def _phrase_term(term: str, max_words: int = 32) -> str:
    words = TOKEN_RE.findall(term.lower())[:max_words]
    if len(words) < 2:
        return ""
    return '"' + " ".join(words) + '"'


def _infer_attribute(value: str) -> str:
    if BUDGET_RE.search(value):
        return "budget"
    if MATERIAL_RE.search(value):
        return "material"
    if COLOR_RE.search(value):
        return "color"
    if SIZE_RE.search(value):
        return "size"
    if STYLE_RE.search(value):
        return "style"
    if USE_CASE_RE.search(value):
        return "use_case"
    return "feature"


def extract_slots(message: str) -> tuple[dict[str, str], float | None]:
    slots: dict[str, str] = {}
    budget_value: float | None = None
    patterns = (
        ("material", MATERIAL_RE), ("color", COLOR_RE), ("size", SIZE_RE),
        ("style", STYLE_RE), ("use_case", USE_CASE_RE), ("feature", FEATURE_RE),
    )
    for attribute, pattern in patterns:
        if match := pattern.search(message):
            slots[attribute] = _clean_value(match.group(1)).lower()
    if match := BRAND_RE.search(message):
        slots["brand"] = _clean_value(match.group(1) or match.group(2) or "")
    if match := BUDGET_RE.search(message):
        raw = next((group for group in match.groups() if group), None)
        if raw:
            try:
                budget_value = float(raw)
                slots["budget"] = f"under ${budget_value:g}"
            except ValueError:
                pass
    if match := CATEGORY_RE.search(message):
        category = _clean_value(match.group(1), 100)
        if category:
            slots["category"] = category
    return slots, budget_value


def _answer_value(message: str) -> str:
    if match := ANSWER_VALUE_RE.search(message):
        return _clean_value(match.group(1))
    if ":" in message and OVERRIDE_RE.search(message):
        return _clean_value(message.rsplit(":", 1)[-1])
    if CATEGORY_RE.search(message) and "." in message:
        tail = _clean_value(message.split(".", 1)[1])
        if tail and not BROWSING_HINT_RE.search(tail):
            return tail
    return ""


def _answer_values(message: str) -> list[str]:
    """Preserve separately disclosed constraints instead of blending phrases."""
    value = _answer_value(message)
    if not value:
        return []
    parts = [_clean_value(part) for part in value.split(";")]
    return _unique_text([part for part in parts if part])


def _call_llm(system: str, user: str, max_tokens: int) -> tuple[str, dict[str, int]] | None:
    """Call Groq or Anthropic and normalize text and token usage."""
    if not LLM_ENABLED:
        return None
    try:
        if LLM_PROVIDER == "groq":
            payload = {
                "model": LLM_MODEL, "max_tokens": max_tokens, "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            request = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"content-type": "application/json", "authorization": f"Bearer {LLM_API_KEY}"},
                method="POST",
            )
        else:
            payload = {
                "model": LLM_MODEL, "max_tokens": max_tokens, "temperature": 0,
                "system": system, "messages": [{"role": "user", "content": user}],
            }
            request = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "content-type": "application/json", "x-api-key": LLM_API_KEY,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
        _debug(f"calling {LLM_PROVIDER} model={LLM_MODEL}")
        with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
        if LLM_PROVIDER == "groq":
            choices = body.get("choices") or []
            message = choices[0].get("message") if choices else {}
            text = message.get("content", "") if isinstance(message, dict) else ""
            usage = body.get("usage") or {}
            normalized = {
                "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            }
        else:
            text = "".join(
                block.get("text", "") for block in body.get("content", [])
                if block.get("type") == "text"
            )
            usage = body.get("usage") or {}
            normalized = {
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
            }
        return text, normalized
    except Exception as error:
        _debug(f"LLM request failed ({error}); using deterministic fallback")
        return None


def _parse_json_block(text: str) -> object | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None


@dataclass
class PreferenceFact:
    attribute: str
    value: str
    hard: bool = False
    turn: int = 0


@dataclass
class SearchPlan:
    action: str
    ask_attribute: str | None
    semantic_query: str
    hard_constraints: list[str] = field(default_factory=list)
    soft_preferences: list[str] = field(default_factory=list)
    excluded_terms: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class RetrievalResult:
    ranked_ids: list[str]
    scores: dict[str, float]
    candidate_count: int
    route_ranks: dict[str, dict[str, int]]


@dataclass
class SessionState:
    profile: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)
    facts: list[PreferenceFact] = field(default_factory=list)
    slots: dict[str, str] = field(default_factory=dict)
    product_features: dict[str, str | None] = field(
        default_factory=lambda: dict.fromkeys(PRODUCT_FEATURE_FIELDS)
    )
    budget_value: float | None = None
    excluded_terms: list[str] = field(default_factory=list)
    asked_attributes: set[str] = field(default_factory=set)
    declined_attributes: set[str] = field(default_factory=set)
    last_asked_attribute: str | None = None
    intent: str | None = None
    turns_seen: int = 0
    questions_asked: int = 0

    def add_fact(self, attribute: str, value: str, turn: int, hard: bool | None = None) -> None:
        value = _clean_value(value)
        if not value or attribute not in ALLOWED_ASK_ATTRIBUTES:
            return
        if hard is None:
            hard = attribute in HARD_ATTRIBUTES
        if attribute in SINGLE_VALUE_ATTRIBUTES:
            self.facts = [fact for fact in self.facts if fact.attribute != attribute]
        key = (attribute, value.casefold())
        if not any((fact.attribute, fact.value.casefold()) == key for fact in self.facts):
            self.facts.append(PreferenceFact(attribute, value, hard, turn))
        self.slots[attribute] = value
        feature_field = QUESTION_TO_FEATURE_FIELD.get(attribute, attribute)
        if feature_field in self.product_features:
            self.product_features[feature_field] = value
        self.declined_attributes.discard(attribute)

    def clear_for_override(self) -> None:
        # The evaluator's override replaces an earlier soft preference with a
        # new hard requirement. Keep facts that were independently stated as
        # hard constraints; erase style/feature/use-case context that may be
        # the superseded preference.
        kept_facts = [
            fact for fact in self.facts if fact.attribute == "category" or fact.hard
        ]
        self.facts = kept_facts
        self.slots = {fact.attribute: fact.value for fact in kept_facts}
        self.product_features = dict.fromkeys(PRODUCT_FEATURE_FIELDS)
        for fact in kept_facts:
            feature_field = QUESTION_TO_FEATURE_FIELD.get(fact.attribute, fact.attribute)
            if feature_field in self.product_features:
                self.product_features[feature_field] = fact.value
        if "budget" not in self.slots:
            self.budget_value = None
        self.declined_attributes.clear()
        self.excluded_terms.clear()
        self.asked_attributes.clear()
        self.questions_asked = 0

    def active_values(self, include_profile: bool = True) -> list[str]:
        values = [fact.value for fact in self.facts]
        if include_profile:
            values.extend(str(value) for value in self.profile.get("preference_tags") or [])
        return _unique_text(values)

    def hard_values(self) -> list[str]:
        return _unique_text([fact.value for fact in self.facts if fact.hard])

    def soft_values(self) -> list[str]:
        return _unique_text([fact.value for fact in self.facts if not fact.hard])


class Agent:
    """Stateful shopping agent with LLM planning and hybrid in-memory retrieval."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._states: dict[str, SessionState] = {}
        self.catalog: dict[str, dict[str, Any]] = {}
        self.search_text: dict[str, str] = {}
        self.search_tokens: dict[str, set[str]] = {}
        self._build_index()
        self.embedding_index = self._load_embedding_index()
        _debug(
            f"ready products={len(self.catalog)} dense="
            f"{'enabled' if self.embedding_index is not None else 'disabled'}"
        )

    @staticmethod
    def _load_embedding_index():
        index_path = Path(LOCAL_EMBEDDING_INDEX)
        if LocalEmbeddingIndex is None or not index_path.is_file():
            _debug(f"dense index unavailable at {index_path}")
            return None
        try:
            return LocalEmbeddingIndex(
                index_path=index_path,
                model_name=LOCAL_EMBEDDING_MODEL,
                expected_model=DEFAULT_MODEL,
            )
        except Exception as error:
            _debug(f"dense index load failed ({error})")
            return None

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "price UNINDEXED, average_rating UNINDEXED, rating_number UNINDEXED, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, ...]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                asin = str(product["parent_asin"])
                document = _text(
                    [
                        product.get("title"), product.get("categories"), product.get("features"),
                        product.get("details"), product.get("store"), product.get("description"),
                    ]
                ).lower()
                self.catalog[asin] = product
                self.search_text[asin] = document
                self.search_tokens[asin] = set(_terms(document))
                batch.append(
                    (
                        asin,
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                        "" if product.get("price") in (None, "") else str(product["price"]),
                        "" if product.get("average_rating") in (None, "") else str(product["average_rating"]),
                        "" if product.get("rating_number") in (None, "") else str(product["rating_number"]),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany(
                        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch
                    )
                    batch.clear()
        if batch:
            cursor.executemany(
                "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch
            )
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._states[session_id] = SessionState(profile=user_profile or {})

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._states.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        state.turns_seen = turn
        self._update_state(state, user_message, turn)

        fallback_plan = self._fallback_plan(state)
        provisional_count = self._estimate_candidate_count(state)
        plan, planner_usage = self._plan_with_llm(state, fallback_plan, provisional_count)
        retrieval = self._retrieve(plan, state, max(1, top_k))

        input_tokens = planner_usage["input_tokens"]
        output_tokens = planner_usage["output_tokens"]
        ranked_ids = retrieval.ranked_ids
        if LLM_ENABLED and LLM_RERANK_ENABLED and ranked_ids:
            reranked, rerank_usage = self._llm_rerank(plan, ranked_ids[:LLM_RERANK_POOL_SIZE])
            input_tokens += rerank_usage["input_tokens"]
            output_tokens += rerank_usage["output_tokens"]
            if reranked:
                selected = set(reranked)
                ranked_ids = reranked + [asin for asin in ranked_ids if asin not in selected]

        ask_attribute = self._choose_ask_attribute(state, plan, retrieval, turn)
        if ask_attribute:
            state.asked_attributes.add(ask_attribute)
            state.last_asked_attribute = ask_attribute
            state.questions_asked += 1
            message = ASK_TEMPLATES[ask_attribute]
        else:
            state.last_asked_attribute = None
            message = (
                f"Here are {min(top_k, len(ranked_ids))} matches based on your active preferences."
                if ranked_ids
                else "I couldn't find a close match yet. What requirement matters most?"
            )
        state.history.append({"role": "assistant", "content": message})
        _debug(
            f"session={session_id} turn={turn} intent={state.intent} candidates="
            f"{retrieval.candidate_count} ask={ask_attribute or 'none'}"
        )
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": asin, "score": round(retrieval.scores.get(asin, 0.0), 6)}
                for asin in ranked_ids[:top_k]
            ],
            "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
        }

    def _update_state(self, state: SessionState, message: str, turn: int) -> None:
        state.history.append({"role": "user", "content": message})
        prior_asked_attribute = state.last_asked_attribute
        is_override = bool(OVERRIDE_RE.search(message))
        if is_override:
            state.clear_for_override()

        if prior_asked_attribute and NO_PREFERENCE_RE.search(message):
            state.declined_attributes.add(prior_asked_attribute)
            feature_field = QUESTION_TO_FEATURE_FIELD.get(prior_asked_attribute, prior_asked_attribute)
            if feature_field in state.product_features:
                state.product_features[feature_field] = "any"

        exclusions = (
            [_clean_value(match.group(1)) for match in EXCLUSION_RE.finditer(message)]
            if not NO_PREFERENCE_RE.search(message)
            else []
        )
        state.excluded_terms = _unique_text(state.excluded_terms + exclusions)
        slots, budget = extract_slots(message)
        for attribute, value in slots.items():
            if any(set(_terms(value)) <= set(_terms(excluded)) for excluded in exclusions):
                continue
            if is_override and any(
                fact.attribute == attribute
                and set(_terms(value)) <= set(_terms(fact.value))
                for fact in state.facts
            ):
                continue
            state.add_fact(attribute, value, turn)
        if budget is not None:
            state.budget_value = budget

        answers = _answer_values(message)
        if answers and not NO_PREFERENCE_RE.search(message):
            for answer in answers:
                attribute = prior_asked_attribute or _infer_attribute(answer)
                if is_override:
                    attribute = _infer_attribute(answer)
                if is_override and any(
                    fact.attribute == attribute
                    and set(_terms(answer)) <= set(_terms(fact.value))
                    for fact in state.facts
                ):
                    continue
                state.add_fact(attribute, answer, turn)

        if any(attribute in HARD_ATTRIBUTES for attribute in slots) or BUYING_HINT_RE.search(message):
            state.intent = "buying"
        elif state.intent is None:
            state.intent = "browsing" if BROWSING_HINT_RE.search(message) else "buying"

    @staticmethod
    def _describe_state(state: SessionState) -> str:
        parts: list[str] = []
        for fact in state.facts:
            label = "hard" if fact.hard else "soft"
            parts.append(f"{fact.attribute} ({label}): {fact.value}")
        tags = state.profile.get("preference_tags") or []
        if tags:
            parts.append("profile preferences: " + ", ".join(str(tag) for tag in tags))
        if state.declined_attributes:
            parts.append("no preference for: " + ", ".join(sorted(state.declined_attributes)))
        if state.excluded_terms:
            parts.append("exclude: " + ", ".join(state.excluded_terms))
        return "; ".join(parts) if parts else "No active product constraints yet"

    def _fallback_plan(self, state: SessionState) -> SearchPlan:
        values = state.active_values(include_profile=False)
        category = state.slots.get("category", "")
        semantic_parts: list[str] = []
        if category:
            semantic_parts.append(f"Product category: {category}")
        non_category = [value for value in values if value.casefold() != category.casefold()]
        if non_category:
            semantic_parts.append("Requirements: " + "; ".join(non_category))
        tags = [str(tag) for tag in state.profile.get("preference_tags") or []]
        if tags:
            semantic_parts.append("General preferences: " + ", ".join(tags))
        semantic_query = ". ".join(semantic_parts) or "clothing shoes or jewelry product"
        return SearchPlan(
            action="ask_and_retrieve",
            ask_attribute=None,
            semantic_query=semantic_query,
            hard_constraints=state.hard_values(),
            # Profile tags remain a weak scoring signal and semantic-query
            # hint. They must not become exact catalog requirements.
            soft_preferences=state.soft_values(),
            excluded_terms=list(state.excluded_terms),
            confidence=0.0,
        )

    def _plan_with_llm(
        self, state: SessionState, fallback: SearchPlan, candidate_count: int
    ) -> tuple[SearchPlan, dict[str, int]]:
        empty_usage = {"input_tokens": 0, "output_tokens": 0}
        if not LLM_ENABLED:
            return fallback, empty_usage
        history = "\n".join(
            f"{item['role'].upper()}: {item['content']}" for item in state.history[-12:]
        )
        system = (
            "You are the planning component of a shopping search agent. Distill only the "
            "currently active request and remove superseded preferences after an override. "
            "Hard constraints are explicit must-have, size, color, material, brand, or budget "
            "requirements. Soft preferences describe use, style, or features. Decide whether "
            "one useful clarification is needed, but retrieval runs either way. Return strict "
            "JSON only with keys action, ask_attribute, semantic_query, hard_constraints, "
            "soft_preferences, excluded_terms, confidence. action is ask_and_retrieve or "
            "retrieve. ask_attribute is null or category, material, color, size, style, brand, "
            "budget, feature, use_case, other. Arrays contain short strings; confidence is 0..1. "
            "Never include product ids."
        )
        user = (
            f"Intent route: {state.intent or 'unknown'}\n"
            f"Deterministic active state: {self._describe_state(state)}\n"
            f"Current lexical candidate count: {candidate_count}\n"
            f"Already asked: {sorted(state.asked_attributes)}\n"
            f"Conversation:\n{history}\nReturn the current search plan."
        )
        result = _call_llm(system, user, max_tokens=500)
        if result is None:
            return fallback, empty_usage
        text, usage = result
        data = _parse_json_block(text)
        if not isinstance(data, dict):
            return fallback, usage
        action = data.get("action")
        if action not in {"ask_and_retrieve", "retrieve"}:
            action = fallback.action
        ask = data.get("ask_attribute")
        if ask not in ALLOWED_ASK_ATTRIBUTES:
            ask = None

        def string_list(name: str, default: list[str]) -> list[str]:
            value = data.get(name)
            if not isinstance(value, list):
                return default
            return _unique_text([
                str(item) for item in value if isinstance(item, (str, int, float))
            ])[:12]

        semantic_query = _clean_value(data.get("semantic_query") or fallback.semantic_query, 700)
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", fallback.confidence))))
        except (TypeError, ValueError):
            confidence = fallback.confidence
        return SearchPlan(
            action=action,
            ask_attribute=ask,
            semantic_query=semantic_query,
            hard_constraints=string_list("hard_constraints", fallback.hard_constraints),
            soft_preferences=string_list("soft_preferences", fallback.soft_preferences),
            excluded_terms=string_list("excluded_terms", fallback.excluded_terms),
            confidence=confidence,
        ), usage

    def _estimate_candidate_count(self, state: SessionState) -> int:
        category = state.slots.get("category", "")
        values = [
            value for value in state.active_values(False)
            if not category or value.casefold() != category.casefold()
        ]
        expression = self._balanced_expression(category, values)
        return self._count_matches(expression, state.budget_value)

    @staticmethod
    def _all_groups_expression(groups: list[str]) -> str:
        fragments = [
            f"({_match_term(group)})" for group in _unique_text(groups) if _match_term(group)
        ]
        return " AND ".join(fragments)

    @staticmethod
    def _balanced_expression(category: str, values: list[str]) -> str:
        category_group = _match_term(category)
        value_groups = [_match_term(value) for value in _unique_text(values) if _match_term(value)]
        optional = " OR ".join(f"({group})" for group in value_groups)
        if category_group and optional:
            return f"({category_group}) AND ({optional})"
        return category_group or optional

    def _retrieve(self, plan: SearchPlan, state: SessionState, top_k: int) -> RetrievalResult:
        category = state.slots.get("category", "")
        state_values = state.active_values(include_profile=False)
        plan_values = _unique_text(plan.hard_constraints + plan.soft_preferences)
        values = _unique_text(state_values + plan_values)
        values_without_category = [
            value for value in values if not category or value.casefold() != category.casefold()
        ]
        route_ranks: dict[str, dict[str, int]] = {}

        def add_route(name: str, expression: str, limit: int = LEXICAL_CANDIDATES) -> None:
            if not expression:
                return
            identifiers = self._fetch_ranked(expression, state.budget_value, limit)
            if identifiers:
                route_ranks[name] = {
                    asin: rank for rank, asin in enumerate(identifiers)
                }

        # Exact phrases are powerful for catalog-derived details and harmless
        # for ordinary paraphrases because the broader routes remain present.
        for index, value in enumerate(values_without_category[:8]):
            phrase = _phrase_term(value)
            if phrase:
                add_route(f"phrase_{index}", phrase, min(LEXICAL_CANDIDATES, 100))

        precision_groups = ([category] if category else []) + values_without_category
        add_route("precision", self._all_groups_expression(precision_groups))
        if len(precision_groups) > 3:
            add_route(
                "recent_precision",
                self._all_groups_expression(precision_groups[:1] + precision_groups[-2:]),
            )
        balanced = self._balanced_expression(category, values_without_category)
        add_route("balanced", balanced)
        broad_groups = [_match_term(value) for value in precision_groups if _match_term(value)]
        add_route("broad", " OR ".join(f"({group})" for group in broad_groups))

        dense: list[tuple[str, float]] = []
        if self.embedding_index is not None and plan.semantic_query:
            try:
                dense = self.embedding_index.search(plan.semantic_query, DENSE_CANDIDATES)
                if dense:
                    route_ranks["dense"] = {
                        asin: rank for rank, (asin, _score) in enumerate(dense)
                    }
            except Exception as error:
                _debug(f"dense query failed ({error})")

        if not route_ranks:
            fallback_ids = self._fetch_ranked("", state.budget_value, LEXICAL_CANDIDATES)
            route_ranks["fallback"] = {
                asin: rank for rank, asin in enumerate(fallback_ids)
            }

        pool: list[str] = []
        for ranks in route_ranks.values():
            pool.extend(ranks)
        pool = list(dict.fromkeys(pool))
        dense_scores = {asin: score for asin, score in dense}
        scores = {
            asin: self._score_candidate(
                asin=asin,
                routes=route_ranks,
                dense_scores=dense_scores,
                category=category,
                values=values_without_category,
                hard_values=plan.hard_constraints,
                excluded_terms=plan.excluded_terms,
                state=state,
            )
            for asin in pool
            if self._within_budget(asin, state.budget_value)
        }
        ranked = sorted(scores, key=lambda asin: (-scores[asin], asin))
        candidate_count = self._count_matches(balanced, state.budget_value) if balanced else len(pool)
        result_size = max(top_k, LLM_RERANK_POOL_SIZE)
        return RetrievalResult(ranked[:result_size], scores, candidate_count, route_ranks)

    def _score_candidate(
        self,
        asin: str,
        routes: dict[str, dict[str, int]],
        dense_scores: dict[str, float],
        category: str,
        values: list[str],
        hard_values: list[str],
        excluded_terms: list[str],
        state: SessionState,
    ) -> float:
        route_weight = {
            "precision": 2.2,
            "recent_precision": 1.7,
            "balanced": 1.3,
            "broad": 0.45,
            # Dense retrieval is a recall route. Exact catalog evidence should
            # dominate once a candidate is in the union.
            "dense": 0.0,
            "fallback": 0.25,
        }
        score = 0.0
        for name, ranks in routes.items():
            rank = ranks.get(asin)
            if rank is None:
                continue
            weight = 2.5 if name.startswith("phrase_") else route_weight.get(name, 0.5)
            score += weight * 20.0 / (20.0 + rank)

        document = self.search_text.get(asin, "")
        document_tokens = self.search_tokens.get(asin, set())
        product = self.catalog.get(asin, {})
        title = _text(product.get("title")).lower()
        categories = _text(product.get("categories")).lower()
        features = _text([product.get("features"), product.get("details")]).lower()
        title_tokens = set(_terms(title))
        category_tokens = set(_terms(categories + " " + title))
        feature_tokens = set(_terms(features))

        requested_category_tokens = set(_terms(category))
        if requested_category_tokens:
            score += 2.4 * len(requested_category_tokens & category_tokens) / len(
                requested_category_tokens
            )

        coverages: list[float] = []
        exact_matches = 0
        normalized_document = " ".join(TOKEN_RE.findall(document))
        for value in values:
            value_tokens = set(_terms(value))
            if not value_tokens:
                continue
            coverage = len(value_tokens & document_tokens) / len(value_tokens)
            coverages.append(coverage)
            normalized_value = " ".join(TOKEN_RE.findall(value.lower()))
            if len(value_tokens) >= 2 and normalized_value in normalized_document:
                exact_matches += 1
            score += 0.6 * len(value_tokens & title_tokens) / len(value_tokens)
            score += 0.8 * len(value_tokens & feature_tokens) / len(value_tokens)
        if coverages:
            score += 3.0 * sum(coverages) / len(coverages)
        score += 2.4 * exact_matches

        for value in hard_values:
            value_tokens = set(_terms(value))
            if value_tokens:
                score += 1.2 * len(value_tokens & document_tokens) / len(value_tokens)
        for excluded in excluded_terms:
            excluded_tokens = set(_terms(excluded))
            if excluded_tokens and excluded_tokens <= document_tokens:
                score -= 8.0

        profile_tokens = set(
            _terms(" ".join(str(tag) for tag in state.profile.get("preference_tags") or []))
        )
        if profile_tokens:
            score += 0.12 * len(profile_tokens & document_tokens) / len(profile_tokens)
        try:
            rating = float(product.get("average_rating") or 0)
            rating_count = float(product.get("rating_number") or 0)
            critical = "critical" in str(state.profile.get("rating_style", "")).lower()
            score += (0.10 if critical else 0.04) * rating
            score += 0.01 * math.log1p(max(0.0, rating_count))
        except (TypeError, ValueError):
            pass
        # Cosine is used to choose which dense-only candidates enter the
        # union. Once present, every product is judged on the same catalog
        # evidence so dense scale cannot displace an exact constraint match.
        return score

    def _choose_ask_attribute(
        self, state: SessionState, plan: SearchPlan, retrieval: RetrievalResult, turn: int
    ) -> str | None:
        if turn >= MAX_TURNS or state.questions_asked >= MAX_CLARIFICATION_QUESTIONS:
            return None

        scores = [retrieval.scores.get(asin, 0.0) for asin in retrieval.ranked_ids[:2]]
        margin = scores[0] - scores[1] if len(scores) == 2 else 0.0
        confident_retrieval = (
            plan.action == "retrieve"
            and plan.confidence >= 0.8
            and retrieval.candidate_count <= 10
            and margin >= 0.5
        )
        if confident_retrieval:
            return None
        if (
            plan.action == "ask_and_retrieve"
            and plan.ask_attribute in ALLOWED_ASK_ATTRIBUTES
            and plan.ask_attribute not in state.asked_attributes
            and plan.ask_attribute not in state.declined_attributes
        ):
            return plan.ask_attribute

        if "category" not in state.slots and "category" not in state.asked_attributes:
            return "category"
        category = state.slots.get("category", "").lower()
        priority = self._questions_by_information_gain(retrieval)
        for attribute in priority:
            if attribute in state.asked_attributes or attribute in state.declined_attributes:
                continue
            if attribute in state.slots and attribute != "feature":
                continue
            if attribute == "size" and category and not any(
                hint in category for hint in SIZE_RELEVANT_HINTS
            ):
                continue
            return attribute
        return None

    def _questions_by_information_gain(self, retrieval: RetrievalResult) -> list[str]:
        """Rank attributes by how well their values divide current candidates."""
        preferred_route = next(
            (
                retrieval.route_ranks[name]
                for name in ("precision", "recent_precision", "balanced", "broad", "dense")
                if name in retrieval.route_ranks
            ),
            {},
        )
        candidate_ids = [
            asin for asin, _rank in sorted(preferred_route.items(), key=lambda item: item[1])[:100]
        ]
        if not candidate_ids:
            candidate_ids = retrieval.ranked_ids[:100]
        scored: list[tuple[float, str]] = []
        for attribute, utility in QUESTION_UTILITY.items():
            buckets = [
                self._candidate_attribute_bucket(self.catalog.get(asin, {}), attribute)
                for asin in candidate_ids
            ]
            known = [bucket for bucket in buckets if bucket]
            distinct = set(known)
            if len(known) < 2 or len(distinct) < 2:
                score = 0.0
            else:
                counts = {value: known.count(value) for value in distinct}
                entropy = -sum(
                    (count / len(known)) * math.log(count / len(known))
                    for count in counts.values()
                )
                normalized_entropy = entropy / math.log(len(distinct))
                coverage = len(known) / max(1, len(buckets))
                score = utility * coverage * normalized_entropy
            scored.append((score, attribute))
        ordered = [
            attribute
            for score, attribute in sorted(scored, key=lambda item: (-item[0], item[1]))
            if score >= 0.2
        ]
        ordered.extend(attribute for attribute in QUESTION_FALLBACK if attribute not in ordered)
        return ordered

    @staticmethod
    def _candidate_attribute_bucket(product: dict[str, Any], attribute: str) -> str:
        text = _text(
            [
                product.get("title"), product.get("features"), product.get("details"),
                product.get("categories"), product.get("description"),
            ]
        )
        if attribute == "brand":
            return _clean_value(product.get("store") or "", 60).casefold()
        if attribute == "budget":
            try:
                price = float(product.get("price"))
            except (TypeError, ValueError):
                return ""
            if price <= 25:
                return "under 25"
            if price <= 50:
                return "25 to 50"
            if price <= 100:
                return "50 to 100"
            if price <= 200:
                return "100 to 200"
            return "over 200"
        patterns = {
            "material": MATERIAL_RE,
            "color": COLOR_RE,
            "size": SIZE_RE,
            "style": STYLE_RE,
            "use_case": USE_CASE_RE,
        }
        if attribute == "feature":
            values = [match.group(1).casefold() for match in FEATURE_RE.finditer(text)]
            values.extend(match.group(0).casefold() for match in CLOSURE_RE.finditer(text))
            # The catalog's feature field is itself a useful partition even
            # when its wording is outside our small normalization vocabulary.
            # Compact signatures measure candidate diversity; they are never
            # exposed as assumed user preferences.
            raw_features = product.get("features") or []
            if not isinstance(raw_features, list):
                raw_features = [raw_features]
            for raw_feature in raw_features[:2]:
                signature_terms = _terms(str(raw_feature))[:6]
                if signature_terms:
                    values.append(" ".join(signature_terms))
        else:
            pattern = patterns.get(attribute)
            if pattern is None:
                return ""
            values = [match.group(1).casefold() for match in pattern.finditer(text)]
        return "|".join(sorted(set(values))[:4])

    def _llm_rerank(
        self, plan: SearchPlan, candidate_ids: list[str]
    ) -> tuple[list[str] | None, dict[str, int]]:
        empty_usage = {"input_tokens": 0, "output_tokens": 0}
        if not candidate_ids:
            return None, empty_usage
        lines: list[str] = []
        for index, asin in enumerate(candidate_ids, 1):
            product = self.catalog.get(asin, {})
            blurb = _clean_value(
                _text(
                    [
                        product.get("title"), product.get("categories"),
                        product.get("features"), product.get("details"), product.get("price"),
                    ]
                ),
                420,
            )
            lines.append(f"{index}. [{asin}] {blurb}")
        system = (
            "Rank catalog candidates for the active shopping plan. Obey hard constraints first, "
            "then semantic fit and soft preferences. Return strict JSON only: an array containing "
            "each supplied parent_asin exactly once, best first. Never invent an id."
        )
        user = (
            f"Semantic request: {plan.semantic_query}\n"
            f"Hard constraints: {plan.hard_constraints}\n"
            f"Soft preferences: {plan.soft_preferences}\n"
            f"Excluded: {plan.excluded_terms}\n\nCandidates:\n" + "\n".join(lines)
        )
        result = _call_llm(system, user, max_tokens=600)
        if result is None:
            return None, empty_usage
        text, usage = result
        order = _parse_json_block(text)
        if not isinstance(order, list):
            return None, usage
        valid = set(candidate_ids)
        ranked: list[str] = []
        for item in order:
            if isinstance(item, str) and item in valid and item not in ranked:
                ranked.append(item)
        return (ranked if ranked else None), usage

    def _within_budget(self, asin: str, budget: float | None) -> bool:
        if budget is None:
            return True
        price = self.catalog.get(asin, {}).get("price")
        if price in (None, ""):
            return True
        try:
            return float(price) <= budget
        except (TypeError, ValueError):
            return True

    def _count_matches(self, expression: str, budget: float | None) -> int:
        if not expression:
            return len(self.catalog)
        sql = "SELECT COUNT(*) FROM products WHERE products MATCH ?"
        params: list[Any] = [expression]
        if budget is not None:
            sql += " AND (price = '' OR CAST(price AS REAL) <= ?)"
            params.append(budget)
        try:
            return int(self.connection.execute(sql, params).fetchone()[0])
        except sqlite3.OperationalError:
            return 0

    def _fetch_ranked(self, expression: str, budget: float | None, limit: int) -> list[str]:
        params: list[Any] = []
        if expression:
            sql = "SELECT parent_asin FROM products WHERE products MATCH ?"
            params.append(expression)
            if budget is not None:
                sql += " AND (price = '' OR CAST(price AS REAL) <= ?)"
                params.append(budget)
            sql += (
                " ORDER BY bm25(products, 0.0, 7.0, 5.0, 3.5, 2.5, 1.5, 1.0, 0.0, 0.0, 0.0) "
                "LIMIT ?"
            )
        else:
            sql = (
                "SELECT parent_asin FROM products ORDER BY "
                "CAST(NULLIF(average_rating, '') AS REAL) DESC, "
                "CAST(NULLIF(rating_number, '') AS REAL) DESC LIMIT ?"
            )
        params.append(limit)
        try:
            return [str(row[0]) for row in self.connection.execute(sql, params).fetchall()]
        except sqlite3.OperationalError:
            return []
