"""Dimensionality reduction helpers for embedding visualization."""

from __future__ import annotations

from typing import Literal, cast

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from design_metrics.utils import FloatArray


def reduce_embeddings_pca(
    embeddings: FloatArray, *, n_components: int = 2
) -> FloatArray:
    """Reduce embeddings using Principal Component Analysis."""

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array.")
    if n_components <= 0:
        raise ValueError("n_components must be positive.")

    reducer = PCA(n_components=n_components, random_state=0)
    return cast(FloatArray, reducer.fit_transform(embeddings))


def reduce_embeddings_tsne(
    embeddings: FloatArray,
    *,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float | Literal["auto"] = "auto",
    random_state: int = 0,
) -> FloatArray:
    """Reduce embeddings using t-distributed stochastic neighbor embedding."""

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array.")

    reducer = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init="pca",
        random_state=random_state,
    )
    return cast(FloatArray, reducer.fit_transform(embeddings))


__all__ = ["reduce_embeddings_pca", "reduce_embeddings_tsne"]
