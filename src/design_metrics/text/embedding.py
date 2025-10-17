"""Embedding utilities for scholarly text using transformer models."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol describing the minimal interface for text embedding models."""

    def encode(
        self, texts: Sequence[str], *, convert_to_numpy: bool = True
    ) -> Sequence[Sequence[float]]:
        """Encode texts into vector representations."""


DEFAULT_MODEL_NAME = "allenai/specter2_base"


def specter2_embed(
    texts: Sequence[str],
    *,
    model: EmbeddingModel | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Generate SPECTER2 embeddings for a collection of texts.

    Args:
        texts: Iterable of textual inputs (titles, abstracts, etc.).
        model: Optionally provide a pre-loaded :class:`SentenceTransformer` to
            avoid re-downloading the weights.
        normalize: If ``True`` the resulting embeddings are L2-normalized.

    Returns:
        A two-dimensional array where each row corresponds to the embedding of a
        single text input.

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.
    """

    runtime_model: EmbeddingModel
    if model is None:
        try:
            module = import_module("sentence_transformers")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "sentence-transformers is required for specter2_embed. Install it via"
                " `pip install sentence-transformers`."
            ) from exc
        model_cls = cast(type[Any], module.SentenceTransformer)
        runtime_model = cast(EmbeddingModel, model_cls(DEFAULT_MODEL_NAME))
    else:
        runtime_model = model

    embeddings = np.asarray(runtime_model.encode(list(texts), convert_to_numpy=True))
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
    return embeddings


__all__ = ["DEFAULT_MODEL_NAME", "specter2_embed"]
