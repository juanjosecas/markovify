import pytest
from markovify.nlp import Corpus, clean_text, SpacyNLP, NLTKNLP


def test_pipeline(tmp_path):
    raw = Corpus(["<p>Hola mundo</p>", "Hola mundo", "", "Otro texto"])
    clean = raw.map(clean_text, html=True).filter_length(2).deduplicate()
    assert clean.texts == ["Hola mundo", "Otro texto"]
    assert len(raw) == 4
    for suffix in (".txt", ".jsonl"):
        path = tmp_path / ("corpus" + suffix)
        clean.save(path)
        assert Corpus.load(path).texts == clean.texts
    assert clean.to_model(state_size=1).make_sentence(test_output=False)


def test_spacy():
    pytest.importorskip("spacy")
    nlp = SpacyNLP(language="es")
    assert nlp.sentences("Hola mundo. Otro texto.") == ["Hola mundo.", "Otro texto."]
    assert nlp.tokenize("Hola, mundo.") == ["Hola", ",", "mundo", "."]
    assert nlp.transform("Hola, mundo.", remove_punctuation=True) == "Hola mundo"
    with pytest.raises(ValueError):
        nlp.transform("Hola", lemmatize=True)


def test_nltk():
    pytest.importorskip("nltk")
    nlp = NLTKNLP()
    assert nlp.tokenize("Hola mundo", preserve_line=True) == ["Hola", "mundo"]
    assert nlp.stem("hablando", preserve_line=True) == "habl"
