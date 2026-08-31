"""Build an offline SentenceTransformers index for the catalog."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from local_embeddings import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_DIR,
    INDEX_SCHEMA_VERSION,
    QUERY_INSTRUCTION,
    catalog_documents,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/catalog_bge_embeddings.npz"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = list(catalog_documents(args.catalog))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("catalog is empty")

    model_source = str(args.model_dir) if args.model_dir.exists() else args.model
    model = SentenceTransformer(model_source)
    args.model_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.model_dir))
    ids = [asin for asin, _ in rows]
    documents = [document for _, document in rows]
    vectors = model.encode(
        documents,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        ids=np.asarray(ids),
        vectors=vectors,
        # Keep `model` for backward compatibility and use model_id for the
        # canonical identity independent of the local directory used to load it.
        model=np.asarray(args.model),
        model_id=np.asarray(args.model),
        query_instruction=np.asarray(QUERY_INSTRUCTION),
        schema_version=np.asarray(INDEX_SCHEMA_VERSION),
    )
    print(f"Wrote {len(ids)} embeddings ({vectors.shape[1]} dimensions) to {args.output}")


if __name__ == "__main__":
    main()
