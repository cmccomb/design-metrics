"""Input and output connectors for design_metrics."""

from design_metrics.io.bibtex import BibEntry, parse_bibtex_entries
from design_metrics.io.pdf import extract_text_from_pdf, iter_pdf_texts

__all__ = [
    "BibEntry",
    "parse_bibtex_entries",
    "extract_text_from_pdf",
    "iter_pdf_texts",
]
