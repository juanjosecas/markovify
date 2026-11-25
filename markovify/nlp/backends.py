"""Thin wrappers around spaCy and NLTK; no automatic downloads."""


class SpacyNLP:
    def __init__(self, model=None, *, language="es", **kwargs):
        import spacy
        self.nlp = spacy.load(model, **kwargs) if model else spacy.blank(language, **kwargs)
        if not any(self.nlp.has_pipe(p) for p in ("parser", "senter", "sentencizer")):
            self.nlp.add_pipe("sentencizer")

    def doc(self, text):
        return self.nlp(text)

    def pipe(self, texts, **kwargs):
        return self.nlp.pipe(texts, **kwargs)

    def tokenize(self, text):
        return [t.text for t in self.nlp.make_doc(text) if not t.is_space]

    def sentences(self, text):
        return [s.text.strip() for s in self.doc(text).sents if s.text.strip()]

    def transform(self, text, *, lemmatize=False, remove_stopwords=False,
                  remove_punctuation=False, remove_numbers=False,
                  keep_pos=None, min_length=0):
        doc = self.doc(text)
        if lemmatize and not doc.has_annotation("LEMMA"):
            raise ValueError("A spaCy model with lemmatization is required")
        if keep_pos is not None and not doc.has_annotation("POS"):
            raise ValueError("A spaCy model with POS tagging is required")
        return " ".join(t.lemma_ if lemmatize else t.text for t in doc
                        if not t.is_space
                        and not (remove_stopwords and t.is_stop)
                        and not (remove_punctuation and t.is_punct)
                        and not (remove_numbers and t.like_num)
                        and (keep_pos is None or t.pos_ in keep_pos)
                        and len(t.text) >= min_length)

    def pos(self, text):
        doc = self.doc(text)
        if not doc.has_annotation("POS"):
            raise ValueError("A spaCy model with POS tagging is required")
        return [(t.text, t.pos_) for t in doc if not t.is_space]

    def entities(self, text):
        doc = self.doc(text)
        if not doc.has_annotation("ENT_IOB"):
            raise ValueError("A spaCy model with NER is required")
        return [(e.text, e.label_, e.start_char, e.end_char) for e in doc.ents]


class NLTKNLP:
    def __init__(self, language="spanish"):
        self.language = language

    def tokenize(self, text, **kwargs):
        from nltk import word_tokenize
        return word_tokenize(text, language=self.language, **kwargs)

    def sentences(self, text):
        from nltk import sent_tokenize
        return sent_tokenize(text, language=self.language)

    def stopwords(self):
        from nltk.corpus import stopwords
        return set(stopwords.words(self.language))

    def remove_stopwords(self, text, **kwargs):
        words = self.stopwords()
        return " ".join(t for t in self.tokenize(text, **kwargs)
                        if t.lower() not in words)

    def stem(self, text, **kwargs):
        from nltk.stem import SnowballStemmer
        stemmer = SnowballStemmer(self.language)
        return " ".join(stemmer.stem(t) for t in self.tokenize(text, **kwargs))
