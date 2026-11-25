import pytest
from markovify import NewlineText


@pytest.mark.parametrize("compiled", [False, True])
def test_constraints(compiled):
    model = NewlineText("La ciencia estudia la materia", state_size=1,
                        random_state=42)
    if compiled:
        model = model.compile()
    opts = dict(test_output=False)
    assert model.make_sentence(start_with="La ciencia estudia", **opts) == "La ciencia estudia la materia"
    assert model.make_sentence(start_with="impossible", **opts) is None
    assert model.make_sentence(end_with="la materia", **opts)
    assert model.make_sentence(include=["ciencia", "materia"], **opts)
    assert model.make_sentence(include=["missing", "materia"], match="any", **opts)
    assert model.make_sentence(include=["missing", "materia"], **opts) is None
    assert model.make_sentence(exclude="materia", **opts) is None
    assert model.make_sentence(include="mater", **opts) is None
    assert model.make_sentence_around("estudia la", **opts)
    assert model.make_sentence_around("la estudia", **opts) is None
    assert len(model.make_sentences(3, unique=False, **opts)) == 3
    assert len(model.make_sentences(3, unique=True, max_attempts=5, **opts)) == 1
    assert model.make_sentences(0, **opts) == []
    assert model.make_sentence(start_with="La ciencia", max_words=1, **opts) is None
    with pytest.raises(ValueError):
        model.make_sentence(match="invalid", **opts)
