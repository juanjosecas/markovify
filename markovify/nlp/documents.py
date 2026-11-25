"""Document extraction with lazy optional dependencies."""
import csv
import json
import os
from pathlib import Path
from .cleaning import strip_html


def _plain(path, encoding, **kwargs):
    return path.read_text(encoding=encoding).splitlines()


def _markup(path, encoding, **kwargs):
    return [strip_html(path.read_text(encoding=encoding))]


def _pdf(path, **kwargs):
    from pypdf import PdfReader
    return [page.extract_text() or "" for page in PdfReader(path).pages]


def _docx(path, **kwargs):
    from docx import Document
    doc = Document(path)
    return [p.text for p in doc.paragraphs] + [
        cell.text for table in doc.tables for row in table.rows for cell in row.cells]


def _odt(path, **kwargs):
    from odf.opendocument import load
    from odf import text, teletype
    doc = load(str(path))
    return [teletype.extractText(p) for p in doc.text.getElementsByType(text.P)]


def _rtf(path, encoding, **kwargs):
    from striprtf.striprtf import rtf_to_text
    return [rtf_to_text(path.read_text(encoding=encoding))]


def _epub(path, **kwargs):
    import ebooklib
    from ebooklib import epub
    book = epub.read_epub(str(path))
    return [strip_html(item.get_content().decode("utf-8"))
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)]


def _records(path, encoding, text_field, **kwargs):
    with path.open(encoding=encoding) as stream:
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in stream if line.strip()]
        else:
            records = json.load(stream)
            if isinstance(records, (str, dict)):
                records = [records]
    if not isinstance(records, list):
        raise ValueError("Expected JSON text or a list of records")
    return [record if isinstance(record, str) else record[text_field]
            for record in records]


def _table(path, encoding, text_field, **kwargs):
    with path.open(encoding=encoding, newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
        if text_field not in (reader.fieldnames or []):
            raise ValueError("Missing text column: " + text_field)
        return [row[text_field] for row in reader]


READERS = {
    ".txt": _plain, ".log": _plain, ".md": _plain, ".markdown": _plain,
    ".html": _markup, ".htm": _markup, ".pdf": _pdf, ".docx": _docx,
    ".odt": _odt, ".rtf": _rtf, ".epub": _epub,
    ".json": _records, ".jsonl": _records, ".csv": _table, ".tsv": _table,
}


def supported_formats():
    return sorted(ext[1:] for ext in READERS)


def discover_files(sources, *, formats=None, recursive=True):
    """Resolve files and directories, deduplicate paths and preserve input order."""
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]
    if isinstance(formats, str):
        formats = [formats]
    allowed = set(READERS) if formats is None else {
        "." + ext.lower().lstrip(".") for ext in formats}
    if allowed - set(READERS):
        raise ValueError("Unsupported formats: " + ", ".join(sorted(allowed - set(READERS))))
    seen = set()
    for source in sources:
        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            candidates = sorted(path.rglob("*") if recursive else path.glob("*"))
        else:
            if path.suffix.lower() not in READERS:
                raise ValueError("Unsupported file: " + str(path))
            candidates = [path]
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in allowed:
                key = candidate.resolve()
                if key not in seen:
                    seen.add(key)
                    yield candidate


def read_document(path, *, encoding="utf-8", text_field="text"):
    path = Path(path)
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError("Unsupported file: " + str(path))
    try:
        entries = reader(path, encoding=encoding, text_field=text_field)
    except ImportError as exc:
        raise ImportError("Install markovify[documents] to read " + path.suffix) from exc
    if not all(isinstance(entry, str) for entry in entries):
        raise ValueError("Document contains non-string text entries")
    return entries
