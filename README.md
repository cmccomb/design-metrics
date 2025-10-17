# design-metrics

A unified metrics library for design research. design-metrics (``design_metrics``) collects
bibliometric, statistical, human-subjects, text analytic, network, and
visualization tools in one modular Python package.

## Installation

```bash
pip install -e .[text]
```

The optional ``text`` extra installs transformer-based embeddings via
``sentence-transformers``.

## Quick start

```python
import design_metrics
from design_metrics.stats import cohen_d

# Effect sizes
d = cohen_d([1, 2, 3], [2, 3, 4])

# Reliability metrics remain under the ``design_metrics.hsr`` namespace
from design_metrics.hsr import cronbach_alpha

alpha = cronbach_alpha([[1, 2, 3], [2, 3, 4], [2, 3, 5]])

# Keyword extraction
from design_metrics.text import rake_keywords

keywords = rake_keywords("Additive manufacturing enables agile prototyping.")

# BibTeX parsing
from design_metrics.io import parse_bibtex_entries

entries = parse_bibtex_entries("""@article{sample,title={Design}}""")

# PDF processing
from design_metrics.io import iter_pdf_texts

for path, text in iter_pdf_texts("/path/to/papers"):
    print(path, text[:80])

# Embedding visualization
import numpy as np
from design_metrics.viz import reduce_embeddings_pca

embeddings = np.random.rand(10, 384)
points = reduce_embeddings_pca(embeddings)
```

## Development

```bash
pip install -e .[text]
pip install -r requirements-dev.txt  # coming soon
pytest
```
