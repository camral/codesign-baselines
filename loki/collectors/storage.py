"""Atomic, resumable numeric shards and a streaming corpus reader."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from enum import Enum
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .encoding import EmbodimentCodec


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return json_value(value.value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def write_json(path, value):
    with open(path, "w") as stream:
        json.dump(json_value(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def file_digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def corpus_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another collector is writing {root}") from error
        try:
            yield root
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class CorpusStore:
    """A renamed shard directory commits its arrays and RNG/counter state together."""

    def __init__(self, root, codec, identity, *, resume):
        self.root, self.codec = Path(root), codec
        self.manifest_path = self.root / "manifest.json"
        self.seen = set()
        self.next_shard = 0
        self.state = None
        if self.manifest_path.exists():
            if not resume:
                raise FileExistsError(f"Corpus already exists: {self.root}; use --resume")
            self.manifest = json.loads(self.manifest_path.read_text())
            if self.manifest["format_version"] != 1:
                raise ValueError("Unsupported corpus format")
            if self.manifest["identity"] != json_value(identity) or self.manifest["schema"] != codec.schema:
                raise ValueError("Resume configuration, source, or embodiment schema differs from this corpus")
            with np.load(self.root / "constants.npz", allow_pickle=False) as constants:
                EmbodimentCodec.from_schema(self.manifest["schema"], constants)
            self._recover()
        else:
            if any(path.name != ".lock" for path in self.root.iterdir()):
                raise FileExistsError(f"Refusing to initialize a nonempty corpus directory: {self.root}")
            (self.root / "shards").mkdir()
            with (self.root / "constants.npz").open("wb") as stream:
                np.savez_compressed(stream, **codec.constants)
                stream.flush()
                os.fsync(stream.fileno())
            self.manifest = {"format_version": 1, "identity": json_value(identity),
                             "schema": codec.schema, "status": "initialized", "shards": 0}
            self.update_manifest(None, "initialized", 0)

    def _recover(self):
        count = 0
        for index, folder in enumerate(sorted((self.root / "shards").iterdir())):
            if folder.name != f"{index:06d}" or not folder.is_dir():
                raise ValueError("Corpus shard sequence is not contiguous")
            metadata = json.loads((folder / "state.json").read_text())
            if file_digest(folder / "data.npz") != metadata["data_sha256"]:
                raise ValueError(f"Corrupt corpus shard: {folder}")
            with np.load(folder / "data.npz", allow_pickle=False) as data:
                ids, hashes = data["ids"], data["hashes"]
                if not np.array_equal(ids, np.arange(count, count + len(ids), dtype=np.int64)):
                    raise ValueError(f"Noncontiguous design IDs in {folder}")
                if len(hashes) != len(ids):
                    raise ValueError(f"Invalid hash count in {folder}")
                for name, field in self.codec.fields.items():
                    if data[name].shape != (len(ids), *field["shape"]) or data[name].dtype != np.dtype(field["dtype"]):
                        raise ValueError(f"Invalid field {name} in {folder}")
                for value in hashes:
                    fingerprint = bytes(value)
                    if fingerprint in self.seen:
                        raise ValueError(f"Duplicate design hash in {folder}")
                    self.seen.add(fingerprint)
                count += len(ids)
            self.state = metadata["state"]
            if self.state["stats"]["accepted"] != count:
                raise ValueError(f"Invalid accepted count in {folder}")
            self.next_shard = index + 1

    def commit(self, rows, hashes, state):
        count = state["stats"]["accepted"]
        arrays = self.codec.stack(rows)
        arrays.update(ids=np.arange(count - len(rows), count, dtype=np.int64),
                      hashes=np.asarray(hashes, dtype="S64"))
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=self.root) as temporary:
            folder = Path(temporary) / "shard"
            folder.mkdir()
            with (folder / "data.npz").open("wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            write_json(folder / "state.json", {"data_sha256": file_digest(folder / "data.npz"), "state": state})
            sync_directory(folder)
            folder.rename(self.root / "shards" / f"{self.next_shard:06d}")
            sync_directory(self.root / "shards")
        self.next_shard += 1
        self.state = json_value(state)

    def update_manifest(self, state, status, target):
        self.manifest.update(status=status, target=target, shards=self.next_shard, state=json_value(state))
        temporary = self.root / "manifest.tmp"
        write_json(temporary, self.manifest)
        temporary.replace(self.manifest_path)
        sync_directory(self.root)


class CorpusReader:
    """Read committed shards one at a time, or reconstruct individual native designs."""

    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["format_version"] != 1:
            raise ValueError("Unsupported corpus format")
        with np.load(self.root / "constants.npz", allow_pickle=False) as constants:
            self.codec = EmbodimentCodec.from_schema(self.manifest["schema"], constants)

    def iter_shards(self, *, verify=True):
        expected = 0
        for index, folder in enumerate(sorted((self.root / "shards").iterdir())):
            if folder.name != f"{index:06d}":
                raise ValueError("Corpus shard sequence is not contiguous")
            metadata = json.loads((folder / "state.json").read_text())
            if verify and file_digest(folder / "data.npz") != metadata["data_sha256"]:
                raise ValueError(f"Corrupt corpus shard: {folder}")
            with np.load(folder / "data.npz", allow_pickle=False) as data:
                arrays = {name: data[name] for name in data.files}
            if not np.array_equal(arrays["ids"], np.arange(expected, expected + len(arrays["ids"]))):
                raise ValueError("Noncontiguous design IDs")
            expected += len(arrays["ids"])
            yield arrays

    def iter_embodiments(self):
        for shard in self.iter_shards():
            for index, design_id in enumerate(shard["ids"]):
                row = {name: shard[name][index] for name in self.codec.fields}
                yield int(design_id), self.codec.decode(row)
