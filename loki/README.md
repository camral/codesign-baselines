# LOKI integration

Fourteen collectors cover 48 supported Codesign Gym presets and all
23 registered Pokémon presets (71 total). The master launcher deduplicates
identical corpus distributions into 36 canonical corpora. Each
inherits from [`BaseCollector`](collectors/base.py). This package implements corpus preparation, Transformer VAE training, frozen latent
clustering, one-cluster PPO/search, and the six-cluster wall-time training and
evaluation pipeline.

For a single command that generates all canonical corpora, see
[the master launcher guide](COLLECT_ALL.md).

## Train the Transformer VAE

After a canonical corpus reaches `status: complete`, train its VAE with:

```bash
python -m loki.train_vae \
  --corpus data/loki/MicrogridRandom \
  --output artifacts/loki/MicrogridRandom/vae \
  --device cuda:0
```

The architecture and core optimizer defaults follow the released LOKI launch:
four Transformer layers, hidden size 32, four attention heads, feed-forward
factor 8, Adam at `1e-4` with `1e-5` weight decay, and KL weight `1e-2`.
LOKI's `scripts/train_vae.sh` uses batch size 4096 and a fixed 200 epochs; its
Python parser separately defaults to batch size 128. This port follows the
released launcher with batch size 4096 and exactly 200 epochs. The KL weight
decays by 0.7 after ten consecutive validation checks without improvement.

The split is deterministic by design ID: IDs divisible by 10 are validation
rows and the remaining 90% are training rows. As in upstream LOKI, validation
samples from the posterior and runs at epochs 0, 10, 20, and so on. Training
keeps the best reconstruction checkpoint using the upstream strict-improvement
rule. Fixed-run completion is recorded separately from diagnostic plateau
status.

Resume an interrupted 200-epoch run:

```bash
python -m loki.train_vae \
  --corpus data/loki/MicrogridRandom \
  --output artifacts/loki/MicrogridRandom/vae \
  --device cuda:0 --resume
```

The output contains `best.pt`, a mean-encoder-only `encoder.pt`, resumable
`last.pt`, `config.json`, per-epoch `history.json`, and `convergence.json`. The
configuration records corpus hashes, exact token schema, split, architecture,
the pinned upstream LOKI revision, and fixed-recipe completion. Production
training refuses an incomplete corpus; `--allow-incomplete` exists only for
pipeline tests and pilots.

Tokenization preserves native structure without inventing limbs: scalar vectors
use one token per component, row-structured arrays retain their rows,
NeroGrasp uses one XYZ token per mutable vertex, and categorical/binary fields
use typed reconstruction heads. See [upstream attribution and adaptations](UPSTREAM.md).

Train all complete canonical corpora across a GPU list, with exactly one VAE
process per GPU:

```bash
python -m loki.train_all_vaes --gpus 0 1 2 3 4 5 6 7
```

The launcher schedules the largest token sequences first, writes each run under
`artifacts/loki/PRESET/vae`, automatically resumes existing checkpoints, skips
completed runs, and records logs plus `training_summary.json`. It uses batch size
4096 except for Warehouse-Congested, which defaults to the measured safe batch
size 1024. Use `--corpora ...` for a subset.

## Form the six frozen clusters

After a corpus VAE has completed its 200 epochs, cluster all 500,000 designs with:

```bash
python -m loki.cluster \
  --corpus data/loki/MicrogridRandom \
  --vae artifacts/loki/MicrogridRandom/vae \
  --output artifacts/loki/MicrogridRandom/clusters \
  --device cuda:0
```

This follows upstream `vae/latent_cluster.py`: it samples one posterior code per
token, flattens the complete token sequence, and calls scikit-learn `KMeans`
with a recorded random seed. The agreed default is six clusters. Latent
extraction streams corpus shards through the GPU into a disk-backed array;
K-means itself runs on the CPU over that full array. The completed artifact
stores centers, one label per design ID, per-cluster design indexes, corpus and
VAE hashes, seeds, estimator parameters, timings, and collapse/size diagnostics.
The temporary latent array is discarded by default; pass `--save-latents` to
retain it. Production runs reject incomplete corpora, unfinished VAE recipes,
mismatched snapshots, empty clusters, and clusters smaller than 128 designs.

Cluster all 36 canonical corpora with one process per GPU:

```bash
python -m loki.cluster_all --gpus 0 1 2 3 4 5 6 7
```

The launcher schedules the longest token sequences first, partitions available
CPU threads across concurrent scikit-learn K-means jobs, and writes logs under
`artifacts/loki/cluster_logs` plus `clustering_summary.json`. It accepts only
VAEs that completed the fixed 200-epoch recipe. Compatible completed cluster
artifacts are skipped; stale or incomplete existing outputs cause a clear
error. Use `--skip-unavailable` to cluster finished VAEs while another VAE is
still training, or `--dry-run` to inspect scheduling without starting jobs.

## Train one cluster controller

Once a canonical cluster artifact exists, run one independent controller with
the released LOKI search defaults:

```bash
python -m loki.single_cluster \
  --env-id Microgrid-Random \
  --cluster-id 0 \
  --output artifacts/loki/MicrogridRandom/online/cluster-0/seed-0 \
  --num-envs 32 \
  --device cuda:0 \
  --seed 0 \
  --iterations 100
```

The preset resolves to its canonical corpus and cluster artifact automatically.
`--corpus` and `--clusters` can override those paths. Resume a coherent
checkpoint with the same arguments plus `--resume`; `--iterations` may be
increased. Every completed PPO iteration saves the policy and optimizer,
observation normalizer, elite pool and return histories, RNG states, artifact
identity, counters, and an atomic latest pointer.

The defaults retain the released online recipe: 20 active embodiments, 2,560
rollout steps, eight PPO epochs, batch size 5,120, a cosine `3e-4` learning
rate with five-iteration warmup, and every two PPO iterations a stochastic
200-step evaluation of 128 candidates followed by literal replacement of the
two worst elites with the two highest-scoring valid candidates. Candidate and
PPO simulator transitions are counted separately. The command is iteration
bounded for isolated verification. The production coordinator below owns the
shared wall-time deadline and candidate batching.

The same path exposes `create_cluster_runtime`, `ClusterRuntime`, and
`EvaluationSelection` for the six-cluster coordinator. Each runtime owns its
policy, optimizer, normalizer, elite state, counters, and isolated NumPy, CPU
Torch, and CUDA RNG streams. Candidate scoring can be chunked and interrupted
at a wall-clock deadline; replacement occurs only after the complete candidate
set has been scored. Cross-cluster selection accepts only synchronized policy
versions and distinct valid designs.

The controller keeps separate actor and critic Transformers with the released
hidden size 128, two heads, five layers, feed-forward width 1,024, learned
positions, zero dropout, and fixed continuous-action standard deviation 0.9.
Codesign observations use one global task-state token plus typed embodiment
tokens from the corpus schema. Candidate scores are the median of completed
episode returns in the fixed horizon. If a Codesign episode does not finish in
that horizon, the score uses its partial return so long-horizon presets remain
rankable. Multiple physical worlds for one logical design are averaged, and a
reset rejection excludes that candidate.

## Run all six clusters online

Run the production LOKI path under the same wall-clock and evaluation contract
as the other baselines:

```bash
python -m loki.gym_train \
  --env-id BallCatcher-Pitch \
  --hours 24 \
  --num-envs 16384 \
  --evaluation-worlds 128 \
  --device cuda:0 \
  --seed 0 \
  --output-dir runs/BallCatcher-Pitch/loki/seed-0
```

`--num-envs` is the total training allocation. It is split exactly and as
evenly as possible across the six controllers; at least six worlds are
required. The corpus and cluster artifact resolve through the canonical preset
mapping. `--corpus` and `--cluster-artifact` override those locations.

Each synchronized cycle advances every controller by one PPO iteration. At the
released search frequency, the coordinator scores 128 candidates and the
current elite pool for every cluster. It commits all six cluster-local
replacements only after every scoring batch completes. The fresh scores rank
the six clusters; the selected cluster's top distinct designs and its one
controller form the next immutable evaluation snapshot. Until the first full
ranking completes, cluster 0 supplies a labeled unranked snapshot.

The one online budget starts at process launch and includes artifact loading,
simulator initialization, PPO, candidate scoring, ranking, checkpoint writes,
and evaluation scheduling. The learning rate uses the released five-update
warmup and cosine shape, with cosine progress measured by elapsed wall time.
Evaluations run in an independent process at 5%, 10%, ..., 95% of the budget,
use deterministic actions, and cannot modify training or selection state.

Training checkpoints under `training_checkpoints/` contain all six policies and
optimizers, six observation normalizers, elite pools and histories, selection
state, isolated RNG states, counters, artifact identities, and a manifest. The
final `result.json` reports physical PPO, candidate, and ranking transitions
separately. The evaluator writes its best self-contained selected-controller
snapshot to `best_checkpoint.pt` and the corresponding designs to
`best_embodiments.npy`.

The standard launchers accept `loki` as a method. They resolve preparation
artifacts with `--loki-corpus-root` and `--loki-artifact-root`; offline VAE or
clustering work never starts inside an online comparison.

Metrics report PPO, candidate-scoring, ranking, initialization, and checkpoint
time separately. Evaluation duration remains labeled as overlapping worker
time. Per-cluster records include controller updates, returns, rejection and
replacement counts, elite turnover, logical and physical worlds, and phase
times. Run-level records include the selected design IDs and ranking age,
training/candidate/evaluation world allocations, physical transition counts,
throughput, peak allocated and reserved CUDA memory, preparation durations,
and corpus, VAE, and clustering hashes.

## Collect and resume

Use the repository's Python environment with Codesign Gym and the requested
simulator installed. Run from the repository root:

```bash
python -m loki.collectors \
  --env-id LQStructure-QP \
  --output-dir data/loki/LQStructure-QP-v0 \
  --device cpu --seed 0 --count 500000
```

`--count` means **accepted unique designs**, not proposal attempts; its default
is 500,000. Use a small count for a pilot. A corpus is generated only when the
collection command or Python method is called.

Resume or extend the corpus with the same command plus `--resume`. The target
includes existing rows. Completed shards restore the counters and proposal RNG.
Changing shard size or increasing the target/attempt limit is allowed. Changing
the preset, seed, sampling configuration, audit cadence, native source fingerprint,
or schema is rejected. Each corpus directory allows one writer. Use a separate
directory for each preset.

BallCatcher, SolarCleaner, and the MuJoCo locomotion collectors accept `--device cuda:0`. NeroGrasp supports `cpu`
or the first visible CUDA device; use `CUDA_VISIBLE_DEVICES` to choose a physical
GPU. The master launcher pins Racing and Warehouse to CUDA ordinal zero (or CPU
with `--device cpu`). Standalone collectors use native JAX backend selection; set
`JAX_PLATFORMS=cpu` before starting Python to force their simulator audits to CPU.
Proposal sampling uses NumPy and does not require controller training.

Use `--env-kwargs '{"name": value}'` for native configuration overrides. Overrides
and resolved native configuration are recorded in the manifest where available.
`python -m loki.collectors --help` lists all supported preset IDs.

## Sampling distributions

| Collector | Proposals |
| --- | --- |
| [BallCatcherCollector](collectors/ball_catcher.py) | Independent uniform values within native per-offset bounds. |
| [LocomotionCollector](collectors/locomotion.py) | Independent uniform values within each task's native normalized morphology-offset bounds. |
| [SolarCleanerCollector](collectors/solar_cleaner.py) | Uniform per-offset values; any leg-presence switch is represented by its sign. |
| [RotateInHandCollector](collectors/rotate_in_hand.py) | Independent uniform values within native bounds: each heightfield pad cell in `[0, 1]`, then each link-offset slot in its clip range. |
| [HandPickUpCollector](collectors/hand_pick_up.py) | The same as RotateInHandCollector. The three presets and RotateInHand deform one hand, so all four share the `HandPickUp-Cube` corpus. |
| [SoftWalkerCollector](collectors/soft_walker.py) | Independent uniform per-cell values within native bounds, then [`common/soft_walker_design.py`](../common/soft_walker_design.py)'s repair (integer kind/channel, largest face-connected body, filled non-manifold junctions): uniform draws are almost never valid bodies. |
| [LQCollector](collectors/lq.py) | Uniform continuous parameters; uniform feasible actuator/sensor counts and uniform subsets. Frozen placements use base values. Native validation rejects mass-budget violations. |
| [MicrogridCollector](collectors/microgrid.py) | Uniform capacities within the preset's load-scaled bounds. |
| [MobileCollector](collectors/mobile.py) | Uniform feasible station count, uniform map positions and dBm power. Native validation enforces power and separation. Inactive rows use `(0, 0, tx_min)`. |
| [RacingCollector](collectors/racing.py) | Uniform values within the five native vehicle-design bounds. |
| [WarehouseCollector](collectors/warehouse.py) | Uniform feasible shelf count, then a uniform subset of eligible cells. No layout-quality or reachability filter is added. |
| [NeroGraspCollector](collectors/nerograsp.py) | Native `MeshSpace.sample(rng=...)`: uniform mutable-vertex perturbations with fixed topology and mounting vertices. |
| [PokemonCollector](collectors/pokemon.py) | Independent uniform species and nature choices for six team members, using `pokemon_common.search_config(preset)`. Native team validation rejects illegal combinations. |

Pokémon stores the benchmark's optimized species/nature choices. Other team
attributes retain the environment's random/default settings; a row does not freeze
every attribute of a compiled battle team. Categorical vocabularies are saved in
the schema, with integer indices in rows and exact support for `None` in early
generations. Corpus reading does not start Showdown. Custom spaces optimizing
moves, items, or stat allocations are not supported by this collector.
Choose the registered preset to select generation and opponent.

Pokémon requires the plugin and a working Node/Showdown installation. This server's
installed package points to a missing Showdown directory; use its source checkout:

```bash
PYTHONPATH=/home/an762/pokemon-codesign-env/src python -m loki.collectors \
  --env-id Pokenv-Random --device cpu --count 100 \
  --output-dir data/loki/PokenvRandom_pilot
```

These are explicit proposal distributions: a uniform shelf count does not make
all warehouse layouts equally likely. Rejection sampling conditions each
distribution on native validity. Collection uses no rewards or controllers.

The LOKI NeroGrasp collector uses **0.0003 m (0.3 mm)** around the template,
overriding the environment's 0.1 mm default. This is independent uniform XYZ noise
on mutable vertices with fixed topology and mountings. A 1,000-proposal
`NeroGrasp-SetScrew` pilot accepted 0.8%; other presets may differ. Override it
explicitly when running a different proposal study:

```bash
python -m loki.collectors \
  --env-id NeroGrasp-SetScrew --device cuda:0 \
  --env-kwargs '{"sample_perturbation_m": 0.0002}' \
  --count 100 --output-dir data/loki/NeroGraspSetScrew_pilot
```

NeroGrasp uses its **full native validator**, including enabled geometry rejectors.
Convex decomposition is deferred to reset and therefore runs only during periodic
reset audits, not for every proposal. It constructs one logical design slot and
the required object worlds for grouped presets.

## Validation and stopping

The base encodes a proposal, decodes exactly the stored numerical values, and
calls native `validate_embodiment`. Every accepted unique row passes that check.
SolarCleaner's output is explicitly interpreted as a rejection mask; other native
validity flags retain their normal polarity. Unexpected backend exceptions
stop collection and surface to the caller.

The first accepted design and every 10,000th accepted design undergo a native
reset audit by default. Reset rejection excludes the candidate. Audits bypass
fallback adapters, so a fallback cannot be stored as the requested design.
`--audit-every 1` audits every candidate that passes validation;
`--audit-every 0` disables audits. Validation does not certify episode performance
or dynamic stability.

`--max-attempts` bounds total proposals across resumed sessions; its default is
`max(10000, 100 * count)`. Collection also stops after
`--max-stalled-attempts` consecutive proposals without a new accepted design
(default 10,000). Reaching either limit records an incomplete status and exits
with code 2. Success exits with 0. Ctrl-C saves buffered progress and exits with
130. The collector never pads a corpus with duplicates to meet a target.

## Files and reading

```text
CORPUS/
  manifest.json          # identity, source hash, schema, status, statistics
  constants.npz          # space bounds; shared mesh vertices and faces
  shards/
    000000/
      data.npz           # ids, hashes, typed field arrays
      state.json         # data checksum, proposal RNG, counters, elapsed time
    000001/
      ...
```

Shards are compressed numeric NumPy archives readable with `allow_pickle=False`.
Fields retain their schema's native precision and integer/binary types. Mesh rows
store only mutable vertices; protected vertices and topology are stored once.
The manifest schema maps array field names to native keys and shapes used by the
VAE tokenizer.

Design IDs are contiguous from zero. SHA-256 hashes deduplicate canonical stored
values, normalizing signed zero and inactive network padding. This is exact
representation deduplication, not a general test of physical equivalence. Memory
usage is one shard of designs plus an O(number of accepted designs) hash set.

Each shard's arrays and RNG/counter checkpoint are committed together by an
atomic directory rename. Resume checks integrity and recovers committed shards
even if the summary manifest is stale. Uncommitted temporary directories are
ignored. A hard kill may discard the current buffered shard; resume restarts at
the last committed proposal state. Status-only checkpoints may have zero rows.

```python
from loki.collectors import CorpusReader, make_collector

reader = CorpusReader("data/loki/LQStructure-QP-v0")
for shard in reader.iter_shards():
    # Every field has shape (batch, *native_leaf_shape).
    ids = shard["ids"]

for design_id, native_design in reader.iter_embodiments():
    # Install native_design in the matching Codesign Gym preset.
    pass

with make_collector("Microgrid-Campus", seed=42, device="cpu") as collector:
    result = collector.collect("data/loki/MicrogridCampus", count=500_000)
    assert result.complete, result.status
```

The manifest retains sampler/schema versions, NumPy/Gymnasium versions, native
configuration and source fingerprints, proposal/acceptance/duplicate/rejection
counts, audits, and cumulative collection time. Collection time starts at
`collect()` and excludes the collector's initial simulator construction; also
record process wall time when measuring total offline preparation cost.

## Verification

```bash
python -m pytest tests/test_loki_collectors.py tests/test_loki_pokemon.py tests/test_loki_vae.py tests/test_loki_cluster.py -q
```

Tests cover registry coverage, resume determinism, stale-manifest recovery,
integrity checks, rejection handling, finite-space exhaustion, writer locking,
interrupted collection, typed serialization, mesh mountings, and native
LQ/Microgrid/Network collection and resets. Small native smoke runs also exercised
Warehouse, Racing, BallCatcher, SolarCleaner, and NeroGrasp with reset auditing.
These checks do not establish large-corpus acceptance or throughput for every preset.

Pokémon checks cover categorical round trips, validation rejection, deterministic
resume, benchmark configuration, and a two-team native Pokenv-Random collection
with reset auditing. VAE checks cover mixed continuous, categorical, and binary tokenization, finite typed losses, convergence artifacts, checkpoint reload, resume, and incomplete-corpus rejection. Large-corpus training throughput and convergence remain unmeasured.
