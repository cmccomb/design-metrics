import numpy as np

from design_metrics.viz.embeddings import reduce_embeddings_pca, reduce_embeddings_tsne


def test_reduce_embeddings_pca_shape() -> None:
    embeddings = np.arange(30).reshape(10, 3)

    reduced = reduce_embeddings_pca(embeddings, n_components=2)

    assert reduced.shape == (10, 2)


def test_reduce_embeddings_tsne_shape() -> None:
    embeddings = np.random.RandomState(0).randn(15, 4)

    reduced = reduce_embeddings_tsne(
        embeddings,
        n_components=2,
        perplexity=5.0,
        random_state=0,
    )

    assert reduced.shape == (15, 2)
