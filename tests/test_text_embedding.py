import numpy as np

from design_metrics.text.embedding import specter2_embed


class DummyModel:
    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> np.ndarray:
        return np.vstack([np.arange(3) + idx for idx, _ in enumerate(texts)])


def test_specter2_embed_with_injected_model() -> None:
    texts = ["title one", "title two"]

    embeddings = specter2_embed(texts, model=DummyModel(), normalize=False)

    assert embeddings.shape == (2, 3)
    assert embeddings[0, 0] == 0


def test_specter2_embed_normalization() -> None:
    texts = ["title one"]

    embeddings = specter2_embed(texts, model=DummyModel(), normalize=True)

    assert np.isclose(np.linalg.norm(embeddings[0]), 1.0)
