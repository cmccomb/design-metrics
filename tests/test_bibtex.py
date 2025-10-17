from design_metrics.io.bibtex import BibEntry, parse_bibtex_entries

SAMPLE_BIBTEX = """
@article{smith2020,
  title={Design Futures},
  author={Smith, Alex and Doe, Jamie},
  year={2020},
  journal={Journal of Design},
}
"""


def test_parse_bibtex_entries_returns_entries() -> None:
    entries = parse_bibtex_entries(SAMPLE_BIBTEX)

    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, BibEntry)
    assert entry.entry_type == "article"
    assert entry.citation_key == "smith2020"
    assert entry.fields["title"] == "Design Futures"


def test_bib_entry_to_dict_roundtrip() -> None:
    entry = parse_bibtex_entries(SAMPLE_BIBTEX)[0]

    as_dict = entry.to_dict()

    assert as_dict["ENTRYTYPE"] == "article"
    assert as_dict["title"] == "Design Futures"
