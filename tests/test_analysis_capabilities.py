from __future__ import annotations

import pandas as pd

import design_metrics as dm


def _sample_tables() -> dict[str, pd.DataFrame]:
    papers = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "title": "Deep Learning for Building Information Modelling",
                "year": 2021,
                "venue": "CAADRIA",
                "abstract": "We explore deep learning techniques for BIM coordination.",
                "keywords": ["deep learning", "BIM"],
            },
            {
                "paper_id": "P2",
                "title": "GAN-assisted Urban Design",
                "year": 2022,
                "venue": "CAAD Futures",
                "abstract": (
                    "Generative adversarial networks enable " "new design workflows."
                ),
                "keywords": "GAN; urban design",
            },
            {
                "paper_id": "P3",
                "title": "Rule-based Planning Systems",
                "year": 2019,
                "venue": "CAADRIA",
                "abstract": "Rule-based approaches complement data-driven methods.",
                "keywords": "rule-based; planning",
            },
            {
                "paper_id": "P4",
                "title": "Deep Learning for Building Information Modeling",
                "year": 2021,
                "venue": "ACADIA",
                "abstract": "A survey of deep learning applications in BIM.",
                "keywords": ["deep learning", "BIM"],
            },
        ]
    )
    authors = pd.DataFrame(
        [
            {"author_id": "A1", "name": "alice tan"},
            {"author_id": "A2", "name": "BOB LEE"},
            {"author_id": "A3", "name": " Carla  Gomez "},
            {"author_id": "A4", "name": "Daniel Ito"},
        ]
    )
    authorships = pd.DataFrame(
        [
            {"paper_id": "P1", "author_id": "A1"},
            {"paper_id": "P1", "author_id": "A2"},
            {"paper_id": "P2", "author_id": "A2"},
            {"paper_id": "P2", "author_id": "A3"},
            {"paper_id": "P3", "author_id": "A4"},
            {"paper_id": "P4", "author_id": "A1"},
        ]
    )
    affiliations = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "author_id": "A1",
                "institution_id": "I1",
                "country": "Singapore",
            },
            {
                "paper_id": "P1",
                "author_id": "A2",
                "institution_id": "I2",
                "country": "Switzerland",
            },
            {
                "paper_id": "P2",
                "author_id": "A2",
                "institution_id": "I3",
                "country": "Japan",
            },
            {
                "paper_id": "P3",
                "author_id": "A4",
                "institution_id": "I4",
                "country": "Australia",
            },
        ]
    )
    references = pd.DataFrame(
        [
            {
                "citing_paper_id": "P2",
                "cited_text": (
                    "Smith, Deep Learning for Building Information Modelling, 2021"
                ),
            },
            {
                "citing_paper_id": "P3",
                "cited_text": "Tan et al., Deep Learning for BIM, 2021",
            },
        ]
    )
    return {
        "papers": papers,
        "authors": authors,
        "authorships": authorships,
        "affiliations": affiliations,
        "references": references,
    }


def test_analysis_capabilities_end_to_end() -> None:
    tables = _sample_tables()

    report = dm.clean.validate_schema(tables)
    assert report.ok, report.to_frame().to_dict(orient="list")

    deduped = dm.clean.dedupe_papers(tables["papers"], similarity=0.9)
    assert len(deduped) == 3

    normalised_authors = dm.clean.normalize_authors(tables["authors"])
    assert normalised_authors.loc[0, "name"] == "Tan, A."

    subset = dm.filter.by_keywords(
        deduped,
        ["deep learning", "gan", "bim"],
        mode="lemma",
    )
    assert set(subset["paper_id"]) == {"P1", "P2"}

    rules = {
        "rules": [
            {"label": "AI", "any": ["deep learning", "gan"]},
            {"label": "Planning", "match": "all", "terms": ["rule", "planning"]},
        ]
    }
    labelled = dm.filter.label_by_rules(deduped, rules)
    assert labelled.loc[labelled["paper_id"] == "P3", "labels"].iloc[0] == ["Planning"]

    trend = dm.metrics.trend(subset, by="year", groupby=["venue"])
    assert {"year", "venue", "count"} <= set(trend.columns)

    top_keywords = dm.metrics.topk(subset, field="keywords", k=2, separator=";")
    assert top_keywords.iloc[0]["keywords"].lower() in {"deep learning", "gan"}

    topic_model = dm.topics.fit(
        text=subset["abstract"],
        model="lda",
        k=2,
        random_state=0,
    )
    topic_summary = dm.topics.describe(topic_model, top_n=5)
    assert len(topic_summary) == 2
    doc_weights = dm.topics.doc_topics(topic_model)
    assert set(doc_weights.columns) == {"document_id", "topic_0", "topic_1"}

    graph = dm.graphs.coauthors(tables["authorships"])
    stats = dm.graphs.stats(graph)
    assert stats["nodes"] >= 4
    communities = dm.graphs.communities(graph)
    assert set(communities.columns) == {"node", "community"}

    geo = dm.geo.aggregate(tables["affiliations"], "country")
    assert not geo.empty

    edges, indegree = dm.refs.in_corpus_citations(
        tables["references"],
        deduped,
        reference_column="cited_text",
        paper_id_column="citing_paper_id",
        corpus_id_column="paper_id",
    )
    assert not edges.empty
    assert indegree.loc[indegree["paper_id"] == "P1", "indegree"].iloc[0] >= 1
