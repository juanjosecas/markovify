import json
import random

import pytest
from markovify import Chain, Text, NewlineText, combine
from markovify.chain import BEGIN, END, GenerationLimitError


def test_seed_and_compile():
    corpus = [["a", "b"], ["a", "c"]]
    a = Chain(corpus, 1, random_state=42)
    b = a.compile()
    assert [a.walk() for _ in range(30)] == [b.walk() for _ in range(30)]


def test_global_rng_unchanged():
    state = random.getstate()
    Chain([["a"]], 1, random_state=42).walk()
    assert random.getstate() == state


def test_generation_limit():
    chain = Chain(None, 1, {(BEGIN,): {"a": 1}, ("a",): {"a": 1}})
    with pytest.raises(GenerationLimitError):
        chain.walk(max_steps=3)
    assert Text(None, state_size=1, chain=chain).make_sentence(max_words=3) is None


@pytest.mark.parametrize("compiled", [False, True])
def test_typed_roundtrip(compiled):
    chain = Chain([[1, ("a", 2), 3.5]], 1, max_steps=20)
    if compiled:
        chain = chain.compile()
    restored = Chain.from_json(chain.to_json())
    assert restored.model == chain.model
    assert restored.max_steps == 20
    assert restored.walk() == [1, ("a", 2), 3.5]


def test_legacy_json():
    raw = json.dumps([[[BEGIN], {"a": 1}], [["a"], {END: 1}]])
    assert Chain.from_json(raw).walk() == ["a"]


def test_text_roundtrip():
    model = NewlineText("a b\na c", state_size=1, well_formed=False,
                        random_state=42, max_steps=20)
    assert isinstance(model.compile(), NewlineText)
    restored = NewlineText.from_json(model.to_json(), random_state=42)
    assert not restored.well_formed
    assert restored.chain.max_steps == 20
    assert [model.make_sentence(test_output=False) for _ in range(20)] == [
        restored.make_sentence(test_output=False) for _ in range(20)
    ]


@pytest.mark.parametrize("weights", [[-1, 2], [0, 0], [float("nan"), 1],
                                     [float("inf"), 1], [True, 1]])
def test_invalid_weights(weights):
    a = Chain([["a"]], 1)
    with pytest.raises(ValueError):
        combine([a, a], weights)


def test_zero_weight():
    a = Chain([["a"]], 1)
    b = Chain([["b"]], 1)
    assert combine([a, b], [1, 0]).model == a.model


@pytest.mark.parametrize("order", [0, -1, True, 1.5])
def test_invalid_order(order):
    with pytest.raises(ValueError):
        Chain([["a"]], order)
