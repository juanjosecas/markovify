"""Optional NLP tools for corpus preprocessing."""
from .cleaning import (clean_text, strip_html, normalize_unicode,
                       normalize_whitespace, remove_controls, remove_urls,
                       remove_emails, remove_accents, lowercase)
from .backends import SpacyNLP, NLTKNLP
from .corpus import Corpus

from .documents import read_document, discover_files, supported_formats
