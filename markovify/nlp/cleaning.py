"""Dependency-free corpus cleaning."""
import re
import unicodedata
from html.parser import HTMLParser


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        elif not self.hidden and tag in ("p", "div", "br", "li", "section"):
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)
        elif not self.hidden and tag in ("p", "div", "li", "section"):
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def strip_html(text):
    parser = _HTMLText()
    parser.feed(text)
    parser.close()
    return "".join(parser.parts)


def normalize_unicode(text, form="NFC"):
    return unicodedata.normalize(form, text)


def normalize_whitespace(text):
    return " ".join(text.split())


def remove_controls(text):
    return "".join(" " if c.isspace() else c for c in text
                   if c.isspace() or unicodedata.category(c) != "Cc")


def remove_urls(text, replacement=" "):
    return re.sub(r"https?://[^\s<>]+|www\.[^\s<>]+", replacement, text)


def remove_emails(text, replacement=" "):
    return re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", replacement, text)


def remove_accents(text):
    return unicodedata.normalize("NFC", "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"))


def lowercase(text):
    return text.lower()


def clean_text(text, *, html=False, urls=False, emails=False,
               accents=False, lower=False, unicode_form="NFC"):
    if html:
        text = strip_html(text)
    if urls:
        text = remove_urls(text)
    if emails:
        text = remove_emails(text)
    text = remove_controls(normalize_unicode(text, unicode_form))
    if accents:
        text = remove_accents(text)
    if lower:
        text = lowercase(text)
    return normalize_whitespace(text)
