#!/usr/bin/env python3
"""Build and query a compact local semantic product store.

This deliberately uses SQLite rather than a server-based vector database: it
keeps the competition submission self-contained, while still persisting
embeddings for dense/hybrid retrieval.  The database is safe to re-run: rows
whose source product text and embedding model have not changed are skipped.

Examples:
  # Add GEMINI_API_KEY=... to .env first.
  python3 scripts/build_vector_store.py build
  python3 scripts/build_vector_store.py query "lightweight waterproof hiking shoe"

The script loads GEMINI_API_KEY from .env (or the environment), uses the
official Google GenAI SDK, and sends one product per embedding API request.
It enforces a strict 100-request-per-minute cap while processing the full
catalog resumably.
"""

from __future__ import annotations

import argparse
import backoff
import hashlib
import json
import math
import os
import sqlite3
import struct
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from google import genai
from google.genai import errors, types
from ratelimit import RateLimitException, limits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data" / "catalog.jsonl"
DEFAULT_DATABASE = ROOT / "data" / "catalog_vectors.sqlite3"
DEFAULT_MODEL = "gemini-embedding-001"
# Gemini Embedding 1 accepts at most 2,048 tokens per input. This conservative
# character limit keeps catalog documents below that limit without a tokenizer.
MAX_EMBEDDING_CHARACTERS = 7_000
REQUESTS_PER_MINUTE = 100
REQUEST_PERIOD_SECONDS = 60
GEMINI_RETRY_SECONDS = 30


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return "; ".join(f"{key}: {as_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return "; ".join(as_text(item) for item in value)
    return str(value)


def product_document(product: dict[str, Any]) -> str:
    """Create one stable, field-labelled embedding document per catalog item."""
    fields = (
        ("Title", product.get("title")),
        ("Brand", product.get("store")),
        ("Categories", product.get("categories")),
        ("Features", product.get("features")),
        ("Description", product.get("description")),
        ("Details", product.get("details")),
    )
    document = "\n".join(f"{name}: {as_text(value)}" for name, value in fields if as_text(value))
    if len(document) <= MAX_EMBEDDING_CHARACTERS:
        return document
    return document[:MAX_EMBEDDING_CHARACTERS].rsplit(" ", 1)[0] + " …"


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding actual environment values."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def content_hash(document: str) -> str:
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            parent_asin TEXT PRIMARY KEY,
            document TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            embedding BLOB NOT NULL,
            dimensions INTEGER NOT NULL,
            model TEXT NOT NULL,
            embedded_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS embeddings_model_idx ON embeddings(model);
        CREATE TABLE IF NOT EXISTS vector_store_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


@backoff.on_exception(backoff.expo, RateLimitException, max_tries=8, jitter=None)
@limits(calls=REQUESTS_PER_MINUTE, period=REQUEST_PERIOD_SECONDS)
def _embed_once(
    client: genai.Client, texts: list[str], model: str, dimensions: int | None, task_type: str
) -> list[list[float]]:
    """Make one rate-limited Gemini request."""
    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=dimensions,
    )
    response = client.models.embed_content(model=model, contents=texts, config=config)
    return [embedding.values for embedding in response.embeddings]


def embed(
    client: genai.Client, texts: list[str], model: str, dimensions: int | None, task_type: str
) -> list[list[float]]:
    """Retry the same batch after Gemini returns a temporary quota rejection."""
    while True:
        try:
            return _embed_once(client, texts, model, dimensions, task_type)
        except errors.ClientError as error:
            if error.code != 429:
                raise
            print(
                f"Gemini stopped us with 429 RESOURCE_EXHAUSTED. "
                f"Waiting {GEMINI_RETRY_SECONDS} seconds before retrying the same batch...",
                flush=True,
            )
            time.sleep(GEMINI_RETRY_SECONDS)


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(blob: bytes, dimensions: int) -> tuple[float, ...]:
    if len(blob) != dimensions * 4:
        raise ValueError("Invalid stored embedding length")
    return struct.unpack(f"<{dimensions}f", blob)


def build(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if not args.catalog.exists():
        raise SystemExit(f"Catalog not found: {args.catalog}")

    args.database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.database)
    try:
        create_schema(connection)
        pending: list[tuple[str, str, str]] = []
        total = skipped = 0
        existing = {
            row[0]: (row[1], row[2])
            for row in connection.execute("SELECT parent_asin, content_hash, model FROM embeddings")
        }
        with args.catalog.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                product = json.loads(line)
                asin = str(product["parent_asin"])
                document = product_document(product)
                digest = content_hash(document)
                if existing.get(asin) == (digest, args.model):
                    skipped += 1
                    continue
                pending.append((asin, document, digest))
                if args.limit and len(pending) + skipped >= args.limit:
                    break
        total = len(pending)
        print(f"{total} products need embeddings; {skipped} already current.")
        if args.dry_run:
            return
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is required. Add it to .env or export it before building the vector store.")

        client = genai.Client(api_key=api_key)
        print(
            f"Rate limit: at most {REQUESTS_PER_MINUTE} embedding API requests per rolling minute "
            "(one product per request)."
        )

        for number, batch in enumerate(chunks(pending, args.batch_size), 1):
            vectors = embed(
                client, [item[1] for item in batch], args.model, args.dimensions, "RETRIEVAL_DOCUMENT"
            )
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding API returned a different number of vectors than requested")
            now = datetime.now(UTC).isoformat()
            rows = [
                (asin, document, digest, pack_vector(vector), len(vector), args.model, now)
                for (asin, document, digest), vector in zip(batch, vectors, strict=True)
            ]
            connection.executemany(
                """INSERT INTO embeddings(parent_asin, document, content_hash, embedding, dimensions, model, embedded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(parent_asin) DO UPDATE SET
                     document=excluded.document, content_hash=excluded.content_hash,
                     embedding=excluded.embedding, dimensions=excluded.dimensions,
                     model=excluded.model, embedded_at=excluded.embedded_at""",
                rows,
            )
            connection.commit()
            print(f"Embedded {min(number * args.batch_size, total)}/{total}", flush=True)
        metadata = {
            "catalog_path": str(args.catalog),
            "model": args.model,
            "requested_dimensions": str(args.dimensions or "native"),
            "last_build_at": datetime.now(UTC).isoformat(),
        }
        connection.executemany(
            "INSERT INTO vector_store_metadata(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            metadata.items(),
        )
        connection.commit()
        count = connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        print(f"Done: {count} embeddings stored in {args.database}")
    finally:
        connection.close()


def query(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is required to embed a search query.")
    if not args.database.exists():
        raise SystemExit(f"Vector store not found: {args.database}. Run the build command first.")
    connection = sqlite3.connect(args.database)
    try:
        row = connection.execute("SELECT model, dimensions FROM embeddings LIMIT 1").fetchone()
        if row is None:
            raise SystemExit("The vector store is empty. Run the build command first.")
        model, dimensions = row
        client = genai.Client(api_key=api_key)
        query_vector = embed(
            client, [args.text], model, dimensions, "RETRIEVAL_QUERY"
        )[0]
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        if query_norm == 0:
            raise SystemExit("Query embedding has zero magnitude")
        scored: list[tuple[float, str, str]] = []
        for asin, document, blob, stored_dimensions in connection.execute(
            "SELECT parent_asin, document, embedding, dimensions FROM embeddings WHERE model = ?", (model,)
        ):
            vector = unpack_vector(blob, stored_dimensions)
            denominator = query_norm * math.sqrt(sum(value * value for value in vector))
            score = sum(left * right for left, right in zip(query_vector, vector)) / denominator if denominator else 0.0
            scored.append((score, asin, document))
        for score, asin, document in sorted(scored, reverse=True)[: args.top_k]:
            title = next((line.removeprefix("Title: ") for line in document.splitlines() if line.startswith("Title: ")), "")
            print(f"{score:.4f}\t{asin}\t{title}")
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("build", build), ("query", query)):
        subparser = subparsers.add_parser(name)
        subparser.set_defaults(handler=handler)
        subparser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    build_parser = subparsers.choices["build"]
    build_parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    build_parser.add_argument("--model", default=DEFAULT_MODEL)
    build_parser.add_argument("--dimensions", type=int, default=768, help="Embedding size; Gemini recommends 768, 1536, or 3072.")
    build_parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Products per API request. Keep this at 1 so the 100-RPM cap maps to 100 products/minute.",
    )
    build_parser.add_argument("--limit", type=int, help="Embed at most this many catalog rows (useful for smoke tests).")
    build_parser.add_argument("--dry-run", action="store_true")
    query_parser = subparsers.choices["query"]
    query_parser.add_argument("text")
    query_parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.handler(arguments)
