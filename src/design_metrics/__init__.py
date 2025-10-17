"""Top-level package for the design-metrics library.

This module exposes key public APIs for convenience while encouraging
modular imports from subpackages like :mod:`design_metrics.hsr` or
:mod:`design_metrics.text`.
"""

from design_metrics.hsr.reliability import cronbach_alpha
from design_metrics.io.bibtex import parse_bibtex_entries
from design_metrics.io.pdf import extract_text_from_pdf, iter_pdf_texts
from design_metrics.stats.effect_sizes import cohen_d, hedges_g
from design_metrics.text.embedding import specter2_embed
from design_metrics.text.keywords import rake_keywords
from design_metrics.viz.embeddings import (
    reduce_embeddings_pca,
    reduce_embeddings_tsne,
)

__all__ = [
    "cohen_d",
    "hedges_g",
    "cronbach_alpha",
    "parse_bibtex_entries",
    "extract_text_from_pdf",
    "iter_pdf_texts",
    "specter2_embed",
    "rake_keywords",
    "reduce_embeddings_pca",
    "reduce_embeddings_tsne",
]
