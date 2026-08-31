"""Optional offline dense retrieval for the frozen product catalog.

The catalog and query must use the same SentenceTransformers model.  The
generated ``.npz`` file is intentionally ignored by git because it is a
reproducible build artifact rather than source data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_INDEX = Path("data/catalog_bge_embeddings.npz")
DEFAULT_MODEL_DIR = Path("data/bge-small-en-v1.5")
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
INDEX_SCHEMA_VERSION = 2


def as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return "; ".join(f"{key}: {as_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return "; ".join(as_text(item) for item in value)
    return str(value)


def product_document(product: dict) -> str:
    fields = (
        ("Title", product.get("title")),
        ("Brand", product.get("store")),
        ("Categories", product.get("categories")),
        ("Features", product.get("features")),
        ("Description", product.get("description")),
        ("Details", product.get("details")),
    )
    return "\n".join(
        f"{name}: {as_text(value)}" for name, value in fields if as_text(value)
    )


def catalog_documents(path: str | Path) -> Iterable[tuple[str, str]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            yield str(product["parent_asin"]), product_document(product)


class LocalEmbeddingIndex:
    """Load a precomputed dense index and perform in-memory cosine search."""

    def __init__(
        self,
        index_path: str | Path,
        model_name: str = DEFAULT_MODEL,
        expected_model: str | None = None,
    ) -> None:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self._np = np
        bundle = np.load(Path(index_path), allow_pickle=False)
        stored_model = str(bundle["model_id"] if "model_id" in bundle else bundle["model"])
        expected = expected_model or model_name
        if not _compatible_model_names(stored_model, expected, model_name):
            raise ValueError(
                f"embedding index uses {stored_model!r}, expected {expected!r}"
            )
        self.ids = [str(value) for value in bundle["ids"].tolist()]
        self.vectors = bundle["vectors"].astype("float32", copy=False)
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.ids):
            raise ValueError(
                "embedding index ids and vector rows do not have matching shapes"
            )
        if not np.isfinite(self.vectors).all():
            raise ValueError("embedding index contains non-finite values")
        self.query_instruction = str(
            bundle["query_instruction"] if "query_instruction" in bundle else QUERY_INSTRUCTION
        )
        # Runtime scoring must not attempt a network request.  The model is
        # downloaded once during index construction and is expected to be
        # present in the local Hugging Face cache for evaluation.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        if Path(model_name).exists():
            model_path = model_name
        else:
            from huggingface_hub import snapshot_download

            # Old index bundles sometimes stored the local model directory in
            # `model`. Resolve the canonical model id instead of attempting to
            # download a path such as data/bge-small-en-v1.5.
            remote_model = expected if "/" in expected else model_name
            model_path = snapshot_download(remote_model, local_files_only=True)
        self.model = SentenceTransformer(model_path)

    def search(self, text: str, top_k: int = 100) -> list[tuple[str, float]]:
        if not text or not self.ids:
            return []
        query = self.model.encode(
            [self.query_instruction + text], normalize_embeddings=True, show_progress_bar=False
        )[0].astype("float32", copy=False)
        scores = self.vectors @ query
        count = min(top_k, len(scores))
        if count == len(scores):
            positions = self._np.argsort(-scores)
        else:
            positions = self._np.argpartition(-scores, count - 1)[:count]
            positions = positions[self._np.argsort(-scores[positions])]
        return [(self.ids[int(pos)], float(scores[int(pos)])) for pos in positions]


def _compatible_model_names(*names: str) -> bool:
    """Treat a canonical model id and its saved local directory as equivalent."""
    cleaned = [str(name).rstrip("/") for name in names if name]
    if not cleaned:
        return False
    basenames = {Path(name).name.casefold() for name in cleaned}
    return len(set(cleaned)) == 1 or len(basenames) == 1
