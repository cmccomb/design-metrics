"""Topic modelling conveniences."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from design_metrics.utils import FloatArray


@dataclass
class TopicModelResult:
    """Structured topic model output for downstream analysis."""

    model_type: str
    topic_term_matrix: FloatArray
    feature_names: Sequence[str]
    doc_topic_matrix: FloatArray
    document_ids: Sequence[str]
    metadata: dict[str, object]


def fit(
    *,
    text: Iterable[str] | pd.Series,
    model: str = "lda",
    k: int = 20,
    max_features: int = 5000,
    random_state: int = 0,
) -> TopicModelResult:
    """Fit a topic model and return a serialisable result."""

    documents, document_ids = _collect_documents(text)
    if not documents:
        raise ValueError("No textual documents supplied for topic modelling")

    if k < 1:
        raise ValueError("k must be at least 1")
    n_topics = min(k, len(documents))
    if n_topics < 1:
        raise ValueError("Not enough documents to fit the requested number of topics")

    model_lower = model.lower()
    if model_lower == "lda":
        vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        matrix = vectorizer.fit_transform(documents)
        lda_model = LatentDirichletAllocation(
            n_components=min(n_topics, matrix.shape[1]),
            learning_method="batch",
            random_state=random_state,
        )
        doc_topic_matrix = lda_model.fit_transform(matrix)
        topic_term_matrix = lda_model.components_
        feature_names = vectorizer.get_feature_names_out()
        metadata: dict[str, object] = {"model": lda_model, "vectorizer": vectorizer}
    elif model_lower == "ctfidf":
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        matrix = vectorizer.fit_transform(documents)
        clusterer = KMeans(
            n_clusters=n_topics,
            random_state=random_state,
            n_init="auto",
        )
        clusterer.fit(matrix)
        topic_term_matrix = clusterer.cluster_centers_
        distances = clusterer.transform(matrix)
        doc_topic_matrix = _normalise_distances(distances)
        feature_names = vectorizer.get_feature_names_out()
        metadata = {"clusterer": clusterer, "vectorizer": vectorizer}
    elif model_lower == "bertopic":  # pragma: no cover - optional dependency
        raise RuntimeError(
            "BERTopic support requires the optional 'bertopic' dependency"
        )
    else:
        raise ValueError(
            "Unsupported model. Choose from 'lda', 'ctfidf', or 'bertopic'."
        )

    topic_term_array = cast(FloatArray, np.asarray(topic_term_matrix, dtype=float))
    doc_topic_array = cast(FloatArray, np.asarray(doc_topic_matrix, dtype=float))

    return TopicModelResult(
        model_type=model_lower,
        topic_term_matrix=topic_term_array,
        feature_names=feature_names,
        doc_topic_matrix=doc_topic_array,
        document_ids=document_ids,
        metadata=metadata,
    )


def describe(model: TopicModelResult, *, top_n: int = 10) -> pd.DataFrame:
    """Return a table describing each topic and its keywords."""

    rows = []
    for index, weights in enumerate(model.topic_term_matrix):
        if weights.size == 0:
            continue
        top_indices = np.argsort(weights)[::-1][:top_n]
        keywords = [
            model.feature_names[idx]
            for idx in top_indices
            if idx < len(model.feature_names)
        ]
        rows.append({"topic": index, "keywords": ", ".join(keywords)})
    return pd.DataFrame(rows)


def doc_topics(model: TopicModelResult) -> pd.DataFrame:
    """Return per-document topic weights."""

    columns = [f"topic_{i}" for i in range(model.doc_topic_matrix.shape[1])]
    frame = pd.DataFrame(model.doc_topic_matrix, columns=columns)
    frame.insert(0, "document_id", model.document_ids)
    return frame


def _collect_documents(text: Iterable[str] | pd.Series) -> tuple[list[str], list[str]]:
    documents: list[str] = []
    ids: list[str] = []
    if isinstance(text, pd.Series):
        for key, value in text.items():
            cleaned = str(value).strip()
            if cleaned:
                documents.append(cleaned)
                ids.append(str(key))
    else:
        for index, value in enumerate(text):
            cleaned = str(value).strip()
            if cleaned:
                documents.append(cleaned)
                ids.append(str(index))
    return documents, ids


def _normalise_distances(distances: FloatArray) -> FloatArray:
    similarities = np.asarray(1.0 / (1.0 + distances), dtype=float)
    row_sums = similarities.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return cast(FloatArray, similarities / row_sums)


__all__ = ["TopicModelResult", "fit", "describe", "doc_topics"]
