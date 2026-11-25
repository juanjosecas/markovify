import json
import gzip

import pytest
from markovify import Chain, Text, NewlineText, save_model, load_model


@pytest.mark.parametrize("suffix,compression", [
    (".json", None), (".json.gz", None), (".h5", None),
    (".hdf5", "gzip"),
])
@pytest.mark.parametrize("kind", [Chain, Text, NewlineText])
@pytest.mark.parametrize("compiled", [False, True])
def test_roundtrip(tmp_path, suffix, compression, kind, compiled):
    if suffix in (".h5", ".hdf5"):
        pytest.importorskip("h5py")
    corpus = [["a", "b"], ["a", "c"]]
    if kind is Chain:
        model = kind(corpus, 1, random_state=42)
    else:
        model = kind(None, state_size=1, parsed_sentences=corpus,
                     random_state=42)
    if compiled:
        model = model.compile()
    chain = getattr(model, "chain", model)
    chain.walk()
    path = tmp_path / ("model" + suffix)
    model.save(path, metadata={"dataset": "example", "n_runs": 2},
               compression=compression)
    restored = load_model(path)
    assert type(restored) is kind
    assert restored.metadata == {"dataset": "example", "n_runs": 2}
    other = getattr(restored, "chain", restored)
    assert other.model == chain.model
    assert other.compiled == chain.compiled
    assert [chain.walk() for _ in range(20)] == [other.walk() for _ in range(20)]
    assert isinstance(kind.load(path), kind)


def test_checksum(tmp_path):
    path = tmp_path / "model.json"
    Chain([["a"]], 1).save(path)
    obj = json.loads(path.read_text())
    obj["payload"]["metadata"]["changed"] = True
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="checksum"):
        load_model(path)


def test_legacy(tmp_path):
    path = tmp_path / "legacy.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump([[["___BEGIN__"], {"a": 1}],
                   [["a"], {"___END__": 1}]], stream)
    assert load_model(path).walk() == ["a"]


def test_atomic_failure(tmp_path, monkeypatch):
    path = tmp_path / "model.json"
    model = Chain([["a"]], 1)
    save_model(model, path)
    original = path.read_bytes()

    def fail(*args):
        raise OSError("simulated failure")

    monkeypatch.setattr("markovify.io.os.replace", fail)
    with pytest.raises(OSError):
        model.save(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_class_mismatch(tmp_path):
    path = tmp_path / "model.json"
    Chain([["a"]], 1).save(path)
    with pytest.raises(TypeError):
        Text.load(path)
