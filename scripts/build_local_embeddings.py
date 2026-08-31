"""Build an offline SentenceTransformers index for the catalog."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from local_embeddings import DEFAULT_MODEL, DEFAULT_MODEL_DIR, catalog_documents


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
    np.savez(args.output, ids=np.asarray(ids), vectors=vectors, model=np.asarray(args.model))
    print(f"Wrote {len(ids)} embeddings ({vectors.shape[1]} dimensions) to {args.output}")


if __name__ == "__main__":
    main()
