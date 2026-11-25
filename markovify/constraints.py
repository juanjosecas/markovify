"""Lexical constraints for bounded text generation."""
from .chain import BEGIN, END


def _tokens(model, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a nonempty string")
    return tuple(model.word_split(value.strip()))


def _terms(model, value):
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    return tuple(_tokens(model, term) for term in value)


def _contains(words, phrase):
    return any(words[i:i + len(phrase)] == phrase
               for i in range(len(words) - len(phrase) + 1))


class LexicalConstraints:
    def __init__(self, model, start_with=None, end_with=None, include=None,
                 exclude=None, include_phrase=None, match="all"):
        if match not in ("all", "any"):
            raise ValueError("match must be 'all' or 'any'")
        self.start = _tokens(model, start_with) if start_with is not None else ()
        self.end = _tokens(model, end_with) if end_with is not None else ()
        self.include = _terms(model, include)
        self.exclude = _terms(model, exclude)
        self.phrases = _terms(model, include_phrase)
        self.match = match

    def accepts(self, words):
        words = tuple(words)
        if self.start and words[:len(self.start)] != self.start:
            return False
        if self.end and words[-len(self.end):] != self.end:
            return False
        if self.include:
            check = all if self.match == "all" else any
            if not check(_contains(words, term) for term in self.include):
                return False
        return (all(_contains(words, term) for term in self.phrases)
                and not any(_contains(words, term) for term in self.exclude))


def prefix_state(chain, prefix):
    """Force a prefix only when every transition has positive support."""
    state = (BEGIN,) * chain.state_size
    for token in prefix:
        if token in (BEGIN, END) or state not in chain.model:
            return None
        options = chain.model[state]
        if chain.compiled:
            choices, cumulative = options
            if token not in choices:
                return None
            i = choices.index(token)
            if cumulative[i] <= (cumulative[i-1] if i else 0):
                return None
        elif options.get(token, 0) <= 0:
            return None
        state = state[1:] + (token,)
    return state
