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
import pandas as pd

import design_metrics as dm

tables = {
    "papers": pd.read_csv("papers.csv"),
    "authors": pd.read_csv("authors.csv"),
    "authorships": pd.read_csv("authorships.csv"),
}

# Validate the adapter output
report = dm.clean.validate_schema(tables)
if not report.ok:
    raise SystemExit(report.to_frame())

# Clean and subset the corpus
papers = dm.clean.dedupe_papers(tables["papers"], similarity=0.9)
authors = dm.clean.normalize_authors(tables["authors"], strategy="lastname_initials")
ai_subset = dm.filter.by_keywords(papers, ["deep learning", "GAN", "BIM"], mode="lemma")

# Trend and keyword summaries
trend = dm.metrics.trend(ai_subset, by="year", groupby=["venue"])
top_keywords = dm.metrics.topk(ai_subset, field="keywords", k=10, separator=";")

# Topic modelling (if abstracts are available)
topic_model = dm.topics.fit(text=ai_subset["abstract"], model="lda", k=10)
topic_summary = dm.topics.describe(topic_model)
doc_topics = dm.topics.doc_topics(topic_model)

# Collaboration network analytics
graph = dm.graphs.coauthors(tables["authorships"])
stats = dm.graphs.stats(graph)
communities = dm.graphs.communities(graph)
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
