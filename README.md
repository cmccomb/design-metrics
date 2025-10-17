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
from design_metrics.stats.effect_sizes import cohen_d

# Effect sizes
d = cohen_d([1, 2, 3], [2, 3, 4])

# Reliability metrics remain under the ``design_metrics.hsr`` namespace
from design_metrics.hsr.reliability import cronbach_alpha

alpha = cronbach_alpha([[1, 2, 3], [2, 3, 4], [2, 3, 5]])

# Keyword extraction
from design_metrics.text.keywords import rake_keywords

keywords = rake_keywords("Additive manufacturing enables agile prototyping.")

# BibTeX parsing
from design_metrics.io.bibtex import parse_bibtex_entries

entries = parse_bibtex_entries("""@article{sample,title={Design}}""")

# PDF processing
from design_metrics.io.pdf import iter_pdf_texts

for path, text in iter_pdf_texts("/path/to/papers"):
    print(path, text[:80])

# Embedding visualization
import numpy as np
from design_metrics.viz.embeddings import reduce_embeddings_pca

embeddings = np.random.rand(10, 384)
points = reduce_embeddings_pca(embeddings)
```

``design_metrics`` is now a namespace package so that future distributions can
provide individual subpackages (for example ``design_metrics.text``) without
requiring the full library. This repository ships all subpackages together, but
you can depend on just the modules you need in your own projects.

## Development

```bash
pip install -e .[text]
pip install -r requirements-dev.txt  # coming soon
pytest
```
