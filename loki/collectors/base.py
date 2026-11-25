"""Shared validation, deduplication, checkpointing, and collection lifecycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import importlib.util
from pathlib import Path
import time

import numpy as np

from .encoding import EmbodimentCodec, array_digest, sample_space
from .storage import CorpusStore, corpus_lock, json_value


def numpy_value(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def source_identity(env_id):
    """Record the installed registry entry and hash its family/core Python sources."""
    import codesign_gym
    from codesign_gym import REGISTRY

    entry, defaults = REGISTRY[env_id]
    module = entry.split(":")[0]
    roots = [Path(importlib.util.find_spec(module).origin).parent,
             Path(importlib.util.find_spec("core.codesign_env").origin).parent]
    digest = hashlib.sha256()
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return {"codesign_gym_version": codesign_gym.__version__, "builder": entry,
            "registry_defaults": defaults, "python_sources_sha256": digest.hexdigest()}


@dataclass(frozen=True)
class CollectionResult:
    output: str
    status: str
    target: int
    stats: dict
    elapsed_seconds: float

    @property
    def complete(self):
        return self.status == "complete"


class BaseCollector(ABC):
    """One native design slot; subclasses own family sampling and API differences.

    Environments supplied by the caller remain caller-owned. The default
    factory creates and closes a native environment, bypassing fallback adapters.
    """

    presets: tuple[str, ...] = ()
    sampler_version = 1
    validator_returns_rejections = False

    def __init__(self, env_id, *, seed=0, device="cuda:0", env_kwargs=None, env=None):
        if env_id not in self.presets:
            raise ValueError(f"{type(self).__name__} supports {self.presets}, received {env_id!r}")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.env_id, self.seed, self.device = env_id, seed, device
        self.env_kwargs = dict(env_kwargs or {})
        self.rng = np.random.default_rng(seed)
        self._owns_env = env is None
        self.env = self.make_env() if env is None else env
        try:
            self.space = self.env.embodiment_space
            self.codec = EmbodimentCodec(self.space)
            self.base_embodiment = (self.env.base_embodiment[0]
                                    if getattr(self.env, "base_embodiment_is_batched", False)
                                    else self.env.base_embodiment)
            self.configure()
        except BaseException:
            self.close()
            raise

    def make_env(self):
        from codesign_gym import make_vec
        env, _ = make_vec(self.env_id, 1, is_eval_env=True, **self.env_kwargs)
        return env

    def configure(self):
        """Read family constraints from the constructed environment."""

    @abstractmethod
    def sample_embodiment(self):
        """Propose one native embodiment using only self.rng."""

    def sample_native_space(self):
        return sample_space(self.space, self.rng)

    def canonicalize(self, design):
        return design

    def validation_input(self, design):
        return design

    def validate_embodiment(self, design):
        """Return (valid, reason); never interpret a rejection mask as validity."""
        try:
            result = numpy_value(self.env.validate_embodiment(self.validation_input(design)))
        except ValueError:
            return False, "validator_value_error"
        if result.dtype != np.bool_ or result.size == 0:
            raise TypeError("Native validate_embodiment must return boolean validity/rejection values")
        valid = not bool(result.any()) if self.validator_returns_rejections else bool(result.all())
        return valid, "" if valid else "native_validation"

    def audit_reset(self, design, design_id):
        # Audit RNG is derived independently; it cannot perturb proposal sampling.
        audit_seed = int(np.random.SeedSequence([self.seed, design_id, 0x10C1]).generate_state(1)[0])
        # SeedSequence emits uint32, but native simulators including
        # BallCatcher store reset seeds in signed int32 state.
        audit_seed &= 0x7FFFFFFF
        _, rejected, _ = self.env.reset(seed=audit_seed, embodiment=self.validation_input(design))
        return not bool(numpy_value(rejected).any())

    def sampler_metadata(self):
        return {}

    def identity(self, audit_every):
        import gymnasium
        cfg = getattr(self.env, "cfg", getattr(self.env, "codesign_config", None))
        return json_value({"collector": type(self).__name__, "sampler_version": self.sampler_version,
                           "env_id": self.env_id, "seed": self.seed, "device": self.device,
                           "env_kwargs": self.env_kwargs,
                           "resolved_config": asdict(cfg) if is_dataclass(cfg) else None,
                           "sampler": self.sampler_metadata(), "audit_every": audit_every,
                           "numpy_version": np.__version__, "gymnasium_version": gymnasium.__version__,
                           "source": source_identity(self.env_id)})

    def collect(self, output, *, count=500_000, shard_size=1024, resume=False,
                max_attempts=None, max_stalled_attempts=10_000, audit_every=10_000,
                progress=None):
        """Collect up to count unique validated designs, with bounded rejection loops.

        count is the desired total including existing rows. max_attempts is a
        total corpus attempt limit; increasing it on resume is supported.
        """
        for name, value in (("count", count), ("shard_size", shard_size),
                            ("max_stalled_attempts", max_stalled_attempts)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(audit_every, int) or isinstance(audit_every, bool) or audit_every < 0:
            raise ValueError("audit_every must be a nonnegative integer")
        max_attempts = max(10_000, count * 100) if max_attempts is None else max_attempts
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        started = time.monotonic()
        with corpus_lock(output) as root:
            store = CorpusStore(root, self.codec, self.identity(audit_every), resume=resume)
            stats = {"attempted": 0, "accepted": 0, "invalid": 0, "duplicates": 0,
                     "errors": 0, "reset_audits": 0, "reset_rejections": 0,
                     "last_accept_attempt": 0, "rejection_reasons": {}}
            previous_elapsed = 0.0
            if store.state is not None:
                self.rng.bit_generator.state = store.state["rng"]
                stats = store.state["stats"]
                previous_elapsed = store.state["elapsed_seconds"]
            else:
                self.rng = np.random.default_rng(self.seed)
            if stats["accepted"] > count:
                raise ValueError("Requested count is smaller than the existing corpus")
            rows, hashes = [], []
            status = "running"
            last_flush = time.monotonic()

            def state():
                return {"stats": stats, "rng": self.rng.bit_generator.state,
                        "elapsed_seconds": previous_elapsed + time.monotonic() - started}

            def flush():
                nonlocal last_flush
                store.commit(rows, hashes, state())
                rows.clear()
                hashes.clear()
                store.update_manifest(state(), status, count)
                last_flush = time.monotonic()
                if progress is not None:
                    progress(json_value({"env_id": self.env_id, "status": status,
                                         "target": count, "stats": stats,
                                         "elapsed_seconds": state()["elapsed_seconds"]}))

            store.update_manifest(state(), status, count)
            try:
                while stats["accepted"] < count:
                    if stats["attempted"] >= max_attempts:
                        status = "attempt_limit"
                        break
                    if stats["attempted"] - stats["last_accept_attempt"] >= max_stalled_attempts:
                        status = "stalled"
                        break
                    stats["attempted"] += 1
                    try:
                        row = self.codec.encode(self.canonicalize(self.sample_embodiment()))
                        fingerprint = array_digest(row).encode("ascii")
                        if fingerprint in store.seen:
                            stats["duplicates"] += 1
                        else:
                            # Validate the decoded stored representation, not a higher-precision proposal.
                            design = self.codec.decode(row)
                            valid, reason = self.validate_embodiment(design)
                            if valid and audit_every and stats["accepted"] % audit_every == 0:
                                stats["reset_audits"] += 1
                                valid = self.audit_reset(design, stats["accepted"])
                                if not valid:
                                    stats["reset_rejections"] += 1
                                    reason = "reset_rejection"
                            if valid:
                                rows.append(row)
                                hashes.append(fingerprint)
                                store.seen.add(fingerprint)
                                stats["accepted"] += 1
                                stats["last_accept_attempt"] = stats["attempted"]
                            else:
                                stats["invalid"] += 1
                                reasons = stats["rejection_reasons"]
                                reasons[reason] = reasons.get(reason, 0) + 1
                    except BaseException:
                        stats["errors"] += 1
                        raise
                    if len(rows) >= shard_size or time.monotonic() - last_flush >= 60:
                        flush()
                if stats["accepted"] == count:
                    status = "complete"
            except KeyboardInterrupt:
                status = "interrupted"
                raise
            except BaseException:
                status = "failed"
                raise
            finally:
                flush()
            return CollectionResult(str(root.resolve()), status, count, json_value(stats), state()["elapsed_seconds"])

    def close(self):
        if self._owns_env and getattr(self, "env", None) is not None:
            self.env.close()
            self.env = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
