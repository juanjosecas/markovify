"""Versioned, atomic model persistence without pickle."""
import gzip
import hashlib
import hmac
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .__version__ import __version__

SCHEMA_VERSION = 1


def _classes():
    from .chain import Chain
    from .text import Text, NewlineText
    return {cls.__name__: cls for cls in (Chain, Text, NewlineText)}


def _json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _digest(value):
    # Preserve transition order, which affects seeded sampling.
    return hashlib.sha256(_json(value)).hexdigest()


def _format(path):
    name = str(path).lower()
    if name.endswith(".json.gz"):
        return "gzip"
    if name.endswith((".h5", ".hdf5")):
        return "hdf5"
    if name.endswith(".json"):
        return "json"
    raise ValueError("Expected .json, .json.gz, .h5 or .hdf5")


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install markovify[hdf5] for HDF5 support") from exc
    return h5py


def _chain(model):
    return getattr(model, "chain", model)


def save_model(model, path, *, metadata=None, compression=None):
    """Save a model atomically. HDF5 compression is None or 'gzip'."""
    path = Path(path)
    fmt = _format(path)
    if compression not in (None, "gzip"):
        raise ValueError("Unsupported compression")
    if fmt == "json" and compression is not None:
        raise ValueError("Use a .json.gz filename for compressed JSON")
    kind = type(model).__name__
    if _classes().get(kind) is not type(model):
        raise TypeError("Unsupported model class")
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")
    chain = _chain(model)
    info = dict(getattr(model, "metadata", {}))
    if metadata is not None:
        info.update(metadata)
    payload = {
        "format": "markovify.model",
        "schema_version": SCHEMA_VERSION,
        "model_class": kind,
        "package_version": __version__,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "metadata": info,
        "model": json.loads(model.to_json()),
        "rng_state": chain.rng.getstate(),
    }
    envelope = {"payload": payload, "sha256": _digest(payload)}
    data = _json(envelope)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".",
                                     dir=path.parent)
    os.close(fd)
    try:
        if fmt == "gzip":
            with gzip.open(temporary, "wb") as stream:
                stream.write(data)
        elif fmt == "hdf5":
            with _h5py().File(temporary, "w") as handle:
                handle.attrs["format"] = "markovify.model"
                handle.attrs["schema_version"] = SCHEMA_VERSION
                handle.create_dataset("envelope", data=bytearray(data),
                                      dtype="u1", compression=compression)
        else:
            Path(temporary).write_bytes(data)
        with open(temporary, "rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_model(path, *, expected_class=None, random_state=None):
    """Load a saved or historical model; verify checksums when available."""
    path = Path(path)
    fmt = _format(path)
    if fmt == "gzip":
        with gzip.open(path, "rb") as stream:
            data = stream.read()
    elif fmt == "hdf5":
        with _h5py().File(path, "r") as handle:
            data = handle["envelope"][()].tobytes()
    else:
        data = path.read_bytes()
    obj = json.loads(data)
    classes = _classes()
    if isinstance(obj, dict) and ("payload" in obj or "sha256" in obj):
        payload = obj["payload"]
        if not hmac.compare_digest(_digest(payload), obj["sha256"]):
            raise ValueError("Model checksum mismatch")
        if payload.get("format") != "markovify.model" or payload.get(
            "schema_version"
        ) != SCHEMA_VERSION:
            raise ValueError("Unsupported persistence schema")
        cls = classes.get(payload["model_class"])
        if cls is None:
            raise ValueError("Unsupported model class")
        raw = payload["model"]
        metadata = payload["metadata"]
        rng_state = payload["rng_state"]
    else:
        # Historical text JSON did not record its concrete subclass.
        cls = classes["Text"] if isinstance(obj, dict) and "chain" in obj else classes["Chain"]
        if cls is classes["Text"] and expected_class is classes["NewlineText"]:
            cls = expected_class
        raw, metadata, rng_state = obj, {}, None
    if expected_class is not None and not issubclass(cls, expected_class):
        raise TypeError("Stored model class does not match requested class")
    model = cls.from_json(json.dumps(raw), random_state=random_state)
    if rng_state is not None and random_state is None:
        version, state, gaussian = rng_state
        _chain(model).rng.setstate((version, tuple(state), gaussian))
    model.metadata = metadata
    return model


class PersistenceMixin:
    """File persistence API shared by chain and text models."""

    def save(self, path, **kwargs):
        return save_model(self, path, **kwargs)

    @classmethod
    def load(cls, path, **kwargs):
        return load_model(path, expected_class=cls, **kwargs)
