from pathlib import Path

import pytest

import design_metrics.io.pdf as pdf_module


class DummyPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class DummyReader:
    def __init__(self, path: Path) -> None:
        self.pages = [DummyPage(f"content of {path.name}")]


def test_extract_text_from_pdf_monkeypatched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dummy_pdf = tmp_path / "paper.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    monkeypatch.setattr(pdf_module, "PdfReader", DummyReader)

    text = pdf_module.extract_text_from_pdf(dummy_pdf)

    assert "content of paper.pdf" in text


def test_iter_pdf_texts_yields_expected_pairs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "nested" / "doc.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(pdf_module, "PdfReader", DummyReader)

    results = list(pdf_module.iter_pdf_texts(tmp_path))

    assert results[0][0].name == "doc.pdf"
    assert "doc.pdf" in results[0][1]
