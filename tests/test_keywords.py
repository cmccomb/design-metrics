from design_metrics.text.keywords import rake_keywords


def test_rake_keywords_returns_expected_phrases() -> None:
    text = "Design research explores collaboration and innovation in architecture."

    keywords = rake_keywords(text, top_k=3)

    assert keywords[0][0] in {
        "design research explores collaboration",
        "design research explores",
    }
    assert all(score > 0 for _, score in keywords)
