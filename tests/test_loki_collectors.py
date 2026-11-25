"""Corpus integrity, native validation contracts, and proposal semantics."""

import json

from gymnasium import spaces
import numpy as np
import pytest

from loki.collectors import COLLECTORS, BaseCollector, CorpusReader, make_collector
from loki.collectors.encoding import EmbodimentCodec, array_digest, sample_space
from loki.collectors.solar_cleaner import SolarCleanerCollector
from loki.collectors.nerograsp import NeroGraspCollector
from loki.collectors.locomotion import LocomotionCollector
from loki.collectors.storage import corpus_lock


class StubEnv:
    def __init__(self, space=None, validate=lambda value: True, reject=lambda value: False):
        self.embodiment_space = space if space is not None else spaces.Box(-1, 1, (3,), dtype=np.float64)
        self.base_embodiment = self.embodiment_space.sample()
        self._validate, self._reject = validate, reject
        self.validated = []
        self.closed = False

    def validate_embodiment(self, design):
        self.validated.append(design)
        return self._validate(design)

    def reset(self, *, seed, embodiment):
        return None, np.array([self._reject(embodiment)]), {}

    def close(self):
        self.closed = True


class StubCollector(BaseCollector):
    presets = ("stub",)

    def sample_embodiment(self):
        return self.sample_native_space()

    def identity(self, audit_every):
        return {"collector": "stub", "seed": self.seed, "audit_every": audit_every}


def hashes(root):
    return np.concatenate([shard["hashes"] for shard in CorpusReader(root).iter_shards()])


def test_nerograsp_loki_sampling_amplitude_is_point_three_mm():
    assert NeroGraspCollector.default_sample_perturbation_m == 0.0003


def test_registry_covers_all_suite_presets():
    from launch_environment_suite import FAMILIES
    from codesign_gym import REGISTRY
    pokemon = {env_id for env_id in REGISTRY if env_id.startswith("Pokenv")}
    expected = {env_id for presets in FAMILIES.values() for env_id in presets}
    expected.update(LocomotionCollector.presets)
    assert set(COLLECTORS) == expected | pokemon
    assert len(set(COLLECTORS.values())) == 14
    assert all(issubclass(cls, BaseCollector) for cls in COLLECTORS.values())


def test_invalid_preset_fails_before_constructing_env():
    with pytest.raises(ValueError, match="Unsupported"):
        make_collector("UnknownPreset")


def test_collection_rejects_invalid_and_reset_fallbacks(tmp_path):
    env = StubEnv(validate=lambda d: np.array([d[0] > -0.4]), reject=lambda d: d[1] < 0)
    result = StubCollector("stub", seed=2, env=env).collect(tmp_path, count=15, shard_size=4, audit_every=1)
    assert result.complete
    assert result.stats["invalid"] > 0 and result.stats["reset_rejections"] > 0
    assert result.stats["attempted"] == result.stats["accepted"] + result.stats["invalid"]
    designs = list(CorpusReader(tmp_path).iter_embodiments())
    assert len(designs) == 15
    for _, design in designs:
        assert design[0] > -0.4 and design[1] >= 0 and design.dtype == np.float64
        assert any(np.array_equal(design, value) for value in env.validated)


def test_resume_matches_uninterrupted_rng_with_different_shard_sizes(tmp_path):
    one, resumed = tmp_path / "one", tmp_path / "resumed"
    StubCollector("stub", seed=7, env=StubEnv()).collect(one, count=25, shard_size=6, audit_every=0)
    StubCollector("stub", seed=7, env=StubEnv()).collect(resumed, count=9, shard_size=4, audit_every=0)
    result = StubCollector("stub", seed=7, env=StubEnv()).collect(
        resumed, count=25, shard_size=8, audit_every=0, resume=True)
    assert result.stats["accepted"] == 25
    np.testing.assert_array_equal(hashes(one), hashes(resumed))


def test_resume_recovers_committed_shard_when_manifest_is_stale(tmp_path):
    StubCollector("stub", env=StubEnv()).collect(tmp_path, count=5, shard_size=2, audit_every=0)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["state"]["stats"]["accepted"] = 0
    manifest["shards"] = 0
    path.write_text(json.dumps(manifest))
    (tmp_path / ".pending-crashed").mkdir()
    result = StubCollector("stub", env=StubEnv()).collect(tmp_path, count=8, audit_every=0, resume=True)
    assert result.complete and len(hashes(tmp_path)) == 8


def test_resume_checks_seed_schema_and_explicit_flag(tmp_path):
    StubCollector("stub", env=StubEnv()).collect(tmp_path, count=2, audit_every=0)
    with pytest.raises(FileExistsError):
        StubCollector("stub", env=StubEnv()).collect(tmp_path, count=2, audit_every=0)
    with pytest.raises(ValueError, match="differs"):
        StubCollector("stub", seed=1, env=StubEnv()).collect(tmp_path, count=3, audit_every=0, resume=True)
    with pytest.raises(ValueError, match="differs"):
        StubCollector("stub", env=StubEnv(spaces.Box(-2, 2, (3,), dtype=np.float64))).collect(
            tmp_path, count=3, audit_every=0, resume=True)


def test_corruption_fails_before_resume_or_read(tmp_path):
    StubCollector("stub", env=StubEnv()).collect(tmp_path, count=2, audit_every=0)
    (tmp_path / "shards" / "000000" / "data.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        list(CorpusReader(tmp_path).iter_shards())
    with pytest.raises(ValueError, match="Corrupt"):
        StubCollector("stub", env=StubEnv()).collect(tmp_path, count=3, audit_every=0, resume=True)


def test_duplicate_only_space_stops_incomplete(tmp_path):
    env = StubEnv(spaces.Box(0, 0, (1,), dtype=np.float32))
    result = StubCollector("stub", env=env).collect(tmp_path, count=2, audit_every=0, max_stalled_attempts=4)
    assert result.status == "stalled"
    assert result.stats["accepted"] == 1 and result.stats["duplicates"] == 4
    assert len(hashes(tmp_path)) == 1


def test_attempt_limit_survives_resume(tmp_path):
    collector = StubCollector("stub", env=StubEnv(validate=lambda value: False))
    result = collector.collect(tmp_path, count=2, audit_every=0, max_attempts=3)
    assert result.status == "attempt_limit" and result.stats["accepted"] == 0
    result = collector.collect(tmp_path, count=2, audit_every=0, max_attempts=5, resume=True)
    assert result.stats["attempted"] == 5


@pytest.mark.parametrize("exception,status", [(RuntimeError("backend failed"), "failed"), (KeyboardInterrupt(), "interrupted")])
def test_unexpected_error_flushes_progress_and_surfaces(tmp_path, exception, status):
    env = StubEnv()
    def validate(value):
        if len(env.validated) == 3:
            raise exception
        return True
    env._validate = validate
    with pytest.raises(type(exception)):
        StubCollector("stub", env=env).collect(tmp_path, count=10, audit_every=0)
    reader = CorpusReader(tmp_path)
    assert reader.manifest["status"] == status
    assert len(list(reader.iter_embodiments())) == 2
    assert reader.manifest["state"]["stats"]["errors"] == 1


def test_exclusive_writer_lock(tmp_path):
    with corpus_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another collector"):
            with corpus_lock(tmp_path):
                pass


def test_solar_validator_reads_true_as_valid():
    # codesign-gym's vector validate_embodiment returns per-lane validity (True = valid) for SolarCleaner too.
    env = StubEnv(validate=lambda value: np.array([True]))
    collector = SolarCleanerCollector("SolarCleaner-Reach", env=env)
    assert collector.validate_embodiment(env.base_embodiment)[0]
    env._validate = lambda value: np.array([False])
    assert not collector.validate_embodiment(env.base_embodiment)[0]
    env._validate = lambda value: np.array([0.0])
    with pytest.raises(TypeError, match="boolean"):
        collector.validate_embodiment(env.base_embodiment)


def test_caller_owned_environment_is_not_closed():
    env = StubEnv()
    with StubCollector("stub", env=env):
        pass
    assert not env.closed


def test_codec_preserves_nested_types_and_precision():
    space = spaces.Dict({"nested": spaces.Dict({"scalar": spaces.Box(-1, 1, (), dtype=np.float64)}),
                         "category": spaces.Discrete(4, start=3), "mask": spaces.MultiBinary(5),
                         "multi": spaces.MultiDiscrete([2, 4], start=[1, 3])})
    codec = EmbodimentCodec(space)
    design = sample_space(space, np.random.default_rng(13))
    design["nested"]["scalar"] = np.asarray(0.12345678912345678, dtype=np.float64)
    row = codec.encode(design)
    restored = codec.decode(row)
    assert space.contains(restored)
    assert restored["nested"]["scalar"] == design["nested"]["scalar"]
    assert isinstance(restored["category"], int)
    assert array_digest(row) == array_digest(codec.encode(restored))


def test_mesh_codec_preserves_mountings_and_saves_faces_once():
    from envs.nerograsp.spaces import Mesh, MeshSpace
    vertices = np.array([[0., 0., 0.], [0.01, 0., 0.], [0., 0.01, 0.], [0., 0., 0.01]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    space = MeshSpace({"tip": Mesh(vertices, faces)}, {"tip": 3})
    codec = EmbodimentCodec(space)
    sample = space.sample(np.random.default_rng(3))
    row = codec.encode(sample)
    assert sum(value.size for value in row.values()) == 3
    restored = EmbodimentCodec.from_schema(codec.schema, codec.constants).decode(row)
    assert space.contains(restored)
    np.testing.assert_array_equal(restored["tip"].vertices, sample["tip"].vertices)
    np.testing.assert_array_equal(restored["tip"].faces, faces)
    sample["tip"].vertices[0, 0] += 1e-9
    with pytest.raises(ValueError, match="mounting"):
        codec.encode(sample)


@pytest.mark.parametrize("env_id", ("LQStructure-LQR", "LQStructure-LQG", "LQStructure-QP",
                                     "Microgrid-Random", "Network-Urban", "Network-Motorway"))
def test_native_small_corpus_validates_and_resets(env_id, tmp_path):
    with make_collector(env_id, seed=5, device="cpu") as collector:
        result = collector.collect(tmp_path, count=6, shard_size=4, audit_every=1)
        assert result.complete
        for _, design in CorpusReader(tmp_path).iter_embodiments():
            assert collector.validate_embodiment(design)[0]
            if env_id.startswith("LQStructure"):
                cfg = collector.cfg
                if not cfg.design_actuators:
                    np.testing.assert_array_equal(design["actuators"], collector.base_embodiment["actuators"])
                if not cfg.design_sensors:
                    np.testing.assert_array_equal(design["sensors"], collector.base_embodiment["sensors"])
            if env_id.startswith("Network"):
                inactive = design["stations"][design["num_bs"]:]
                assert np.all(inactive == [0, 0, collector.cfg.tx_min])


@pytest.mark.parametrize("env_id", ("Warehouse-Small", "Warehouse-Medium", "Warehouse-Congested"))
def test_warehouse_sampler_covers_counts_and_native_constraints(env_id):
    from codesign_gym import REGISTRY
    from envs.robot_warehouse.config import make_config
    from envs.robot_warehouse.embodiment import make_embodiment_space, validate_embodiment
    cfg = make_config(REGISTRY[env_id][1]["setup"])
    env = StubEnv(make_embodiment_space(cfg), validate=lambda d: validate_embodiment(d, cfg))
    env.cfg = cfg
    with make_collector(env_id, env=env) as collector:
        counts = set()
        for _ in range(100):
            design = collector.sample_embodiment()
            assert collector.validate_embodiment(design)[0]
            counts.add(int(design.sum()))
        assert len(counts) > 10
