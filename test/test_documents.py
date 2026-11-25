import json
import pytest
from markovify.nlp import Corpus, discover_files, read_document


def test_discovery(tmp_path):
    (tmp_path / "a.TXT").write_text("one\ntwo", encoding="utf-8")
    (tmp_path / "b.html").write_text("<p>three</p>", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.md").write_text("four", encoding="utf-8")
    (sub / "ignore.bin").write_bytes(b"abc")
    corpus = Corpus.load([tmp_path, tmp_path / "a.TXT"])
    assert len(corpus) == 4
    assert len(corpus.sources) == 4
    assert len(corpus.load_report) == 3
    assert Corpus.load(tmp_path, formats=["TXT", ".md"]).texts == ["one", "two", "four"]
    assert len(list(discover_files(tmp_path, recursive=False))) == 2


def test_records_and_errors(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([{"body": "one"}]))
    (tmp_path / "b.csv").write_text("body\ntwo\n")
    (tmp_path / "c.json").write_text("invalid")
    corpus = Corpus.load(tmp_path, text_field="body", errors="skip")
    assert corpus.texts == ["one", "two"]
    assert corpus.load_report[-1]["status"] == "error"
    with pytest.raises(ValueError):
        Corpus.load(tmp_path, text_field="body")


def test_docx(tmp_path):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_paragraph("Hello world")
    path = tmp_path / "sample.docx"
    doc.save(path)
    assert read_document(path) == ["Hello world"]


def test_pdf_empty(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    path = tmp_path / "blank.pdf"
    with path.open("wb") as stream:
        writer.write(stream)
    assert Corpus.load(path).load_report[0]["status"] == "empty"
