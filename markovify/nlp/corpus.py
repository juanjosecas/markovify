"""Composable corpus transformations and text persistence."""
import json
from pathlib import Path
from .documents import discover_files, read_document


class Corpus:
    def __init__(self, texts):
        self.texts = [texts] if isinstance(texts, str) else list(texts)
        if not all(isinstance(t, str) for t in self.texts):
            raise TypeError("Corpus entries must be strings")

    def __iter__(self):
        return iter(self.texts)

    def __len__(self):
        return len(self.texts)

    @classmethod
    def load(cls, path, *, formats=None, recursive=True, encoding="utf-8",
             text_field="text", errors="raise"):
        """Load a file, directory or iterable of paths with format detection."""
        if errors not in ("raise", "skip"):
            raise ValueError("errors must be 'raise' or 'skip'")
        texts, sources, report = [], [], []
        for file in discover_files(path, formats=formats, recursive=recursive):
            try:
                entries = read_document(file, encoding=encoding, text_field=text_field)
                texts.extend(entries)
                sources.extend([str(file)] * len(entries))
                report.append(dict(path=str(file), entries=len(entries), status="ok" if any(t.strip() for t in entries) else "empty"))
            except Exception as exc:
                if errors == "raise":
                    raise
                report.append(dict(path=str(file), entries=0, status="error", error=str(exc)))
        corpus = cls(texts)
        corpus.sources = sources
        corpus.load_report = report
        return corpus

    def save(self, path, *, encoding="utf-8", text_field="text"):
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in (".txt", ".jsonl"):
            raise ValueError("Expected .txt or .jsonl")
        if suffix == ".txt" and any("\n" in t or "\r" in t for t in self):
            raise ValueError("TXT requires one entry per line; use JSONL")
        with path.open("w", encoding=encoding) as stream:
            for text in self:
                stream.write((json.dumps({text_field: text}, ensure_ascii=False)
                              if suffix == ".jsonl" else text) + "\n")
        return path

    def map(self, function, **kwargs):
        return type(self)(function(t, **kwargs) for t in self)

    def split(self, function, **kwargs):
        return type(self)(part for t in self for part in function(t, **kwargs))

    def filter(self, predicate):
        return type(self)(t for t in self if predicate(t))

    def filter_length(self, min_words=1, max_words=None):
        if min_words < 0 or (max_words is not None and max_words < min_words):
            raise ValueError("Invalid word limits")
        return self.filter(lambda t: len(t.split()) >= min_words and
                           (max_words is None or len(t.split()) <= max_words))

    def deduplicate(self, *, case_sensitive=True):
        seen = set()
        result = []
        for text in self:
            key = text if case_sensitive else text.casefold()
            if key not in seen:
                seen.add(key)
                result.append(text)
        return type(self)(result)

    def to_text(self, separator="\n"):
        return separator.join(self)

    def to_model(self, *, tokenizer=None, **kwargs):
        from ..text import NewlineText
        tokenize = tokenizer or str.split
        return NewlineText(None, parsed_sentences=[tokenize(t) for t in self if t.strip()],
                           **kwargs)
