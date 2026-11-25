# CoDesign Gym Baselines

Reusable implementations of optimization and reinforcement-learning approaches
for robot co-design. The adapters in this repository are environment-agnostic:
they expose importable problem factories instead of assuming the simulators used
in the original papers.

## Included methods

| Baseline | Method | Location |
| --- | --- | --- |
| FastTD3 | Off-policy reinforcement learning | [`fasttd3_rl/`](fasttd3_rl/) |
| PPO + NGOpt | Independent policy and embodiment optimization | [`ppo_ngopt_rlgo/`](ppo_ngopt_rlgo/) |
| CMA-ES | Joint evolutionary design and policy search | [`cmaes_evolutionary/`](cmaes_evolutionary/) |
| FastTD3 + CMA-ES | FastTD3 controller with CMA-ES embodiment search | [`fasttd3_cmaes/`](fasttd3_cmaes/) |
| LOKI | Latent embodiment clustering with cluster-specific Transformer PPO controllers | [`loki/`](loki/) |

## Installation

The complete reusable stack uses one Python 3.12 environment and one dependency
file. Create the environment and install from the repository root:

```bash
git clone <repository-url>
cd codesign-baselines

conda create -n codesign-baselines python=3.12 -y
conda activate codesign-baselines
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# The environments: Codesign Gym, here checked out next to this repository.
git clone https://github.com/camral/codesign-gym.git ../codesign-gym
python -m pip install -e ../codesign-gym
```

With uv instead of pip (`uv pip install`), add
`--index-strategy unsafe-best-match --no-sources` to both install commands. By
default uv takes each package only from the first index that has it, and the
PyTorch index also carries old releases of packages such as `tqdm` and `numpy`.
Without `--no-sources`, uv also follows the mujoco_warp fork's own
`tool.uv.sources`, which fetch MuJoCo from its nightly index: the pinned release
then fails to resolve, or a nightly build is installed instead.

`requirements.txt` also installs FastTD3 as an editable local package.

To install only what the FastTD3 + CMA-ES trainer needs into an existing Codesign Gym
environment, install this repository as a package instead (with uv, add `--no-sources`):

```bash
python -m pip install -e .
```

It leaves out Codesign Gym, MuJoCo and mujoco_warp, so it keeps the mujoco_warp fork in place.
After it, `python -m fasttd3_cmaes.gym_train` runs from any directory.

On Linux, the JAX dependency selects CUDA 13; use a compatible NVIDIA driver. Simulator packages required by a new environment belong to that
environment and can be installed into the same Conda environment afterward.

### The mujoco_warp fork

The `RotateInHand-Cube` and `HandPickUp-*` environments deform a heightfield pad
differently in every world, which upstream mujoco_warp cannot represent.
`requirements.txt` therefore installs mujoco_warp from a fork's `integration_gjk_fix`
branch (matching Codesign Gym's `morphhand` extra),
and pins MuJoCo to the release it was tested with. Every other environment runs unchanged on the fork.

If upstream `mujoco-warp` ends up installed over it, the hand environments fail
with this instruction to restore the fork:

```bash
python -m pip install --no-deps \
  "mujoco-warp @ git+https://github.com/y-tsvetkov/mujoco_warp.git@integration_gjk_fix"
```

The pinned legacy stacks used only to reproduce the original paper experiments
(IsaacGym, HumanoidBench and MuJoCo Playground) are not
part of this setup.

### GPU and experiment logging

Every reusable entry point accepts `--device`. Use `auto` (the default),
`cpu`, `cuda`, an indexed CUDA device such as `cuda:1`, or `mps`. Passing
`--wandb-project PROJECT` (also accepted as `--wandb_project PROJECT`) enables
Weights & Biases logging to that project.

Environment/problem factories that declare a `device` keyword receive the
resolved device string. CMA-ES evaluators and NGOpt objectives may likewise
declare `device`; optimization bookkeeping remains on CPU while expensive
simulation and policy evaluation can run on the selected accelerator.

## Quick starts

Run commands from the repository root unless a section says otherwise.

### CMA-ES

Use Nevergrad CMA-ES to optimize bounded continuous design and policy variables:

```bash
python -m cmaes_evolutionary.main --budget 100 \
  --device cuda:0 --wandb-project my-project
```

Defaults are in [`cmaes_evolutionary/config/default.json`](cmaes_evolutionary/config/default.json).
Use `--config PATH` or command-line options to change dimensions, bounds, budget,
evaluation repeats, and seed. A custom evaluator receives design and policy
arrays and an optional `device` keyword, and returns a scalar reward (or a cost
when `maximize` is false).

### PPO + NGOpt

The PPO controller and NGOpt design search can run independently:

```bash
python -m ppo_ngopt_rlgo ppo train --device cuda:0 --wandb-project my-project
python -m ppo_ngopt_rlgo ppo evaluate --device cuda:0 --wandb-project my-project
python -m ppo_ngopt_rlgo ngopt optimize --device cuda:0 --wandb-project my-project
```

The checked-in CartPole and quadratic-objective configurations are smoke tests.
Replace their environment and objective hooks for a real co-design problem. See
[`ppo_ngopt_rlgo/README.md`](ppo_ngopt_rlgo/README.md) for the configuration and
design-policy exchange interfaces.

### FastTD3

FastTD3 accepts Gymnasium environments with finite continuous `Box` actions and
`Box`, `Dict`, or `Tuple` observations:

```bash
python -m fast_td3.codesign.train \
  --env-name MyCoDesign-v0 \
  --env-module my_project.environments \
  --device cuda:0 \
  --wandb-project my-project \
  --num-envs 16 \
  --num-eval-envs 4 \
  --v-min -100 \
  --v-max 100
```

Use `--env-factory module:callable` for a zero-argument factory and
`--env-kwargs` for JSON construction arguments. FastTD3 rescales actions from
`[-1, 1]` to the environment's bounds. Run
`python -m fast_td3.codesign.train --help` for all options.

### FastTD3 + CMA-ES

FastTD3 learns an embodiment-conditioned controller while CMA-ES searches the
embodiment alone. The schedule follows PPO + NGOpt: a FastTD3 training phase on
the current designs, then one CMA-ES generation scored by the frozen
deterministic actor, one complete episode per world. Scoring transitions also
enter the replay buffer, because FastTD3 is off-policy. The evaluator receives
the best accepted design of the latest generation with the current policy, so
the design and policy it sees always come from the same point in training.

```bash
python -m fasttd3_cmaes.gym_train \
  --env-id BallCatcher-Pitch \
  --hours 3 \
  --num-envs 16384 \
  --population-size 128 \
  --device cuda:0 \
  --output-dir exp_results/ballcatcher-pitch-fasttd3-cmaes \
  --project my-project
```

`--search-frequency N` sets the FastTD3 vector steps (each followed by the
FastTD3 gradient updates) before every CMA-ES generation. By default a phase
lasts until every world has finished at least one episode. LOKI's flag of the
same name counts PPO updates instead. FastTD3 hyperparameters and replay options
are the `fasttd3` baseline's; `--population-size` and `--covariance` are the
`cmaes` baseline's. The CMA-ES design optimiser is in
[`fasttd3_cmaes/design_optimiser.py`](fasttd3_cmaes/design_optimiser.py); the
joint CMA-ES baseline is unchanged.

### LOKI

The LOKI integration is a Codesign Gym port of
[Convergent Functions, Divergent Forms](https://arxiv.org/abs/2505.21665). It
uses the released implementation at commit
`24aaac7eb2ecd8f1aada1e1ed0e2d91577fb578a` as its pinned reference. The port
has four stages:

1. Generate a reusable offline corpus of 500,000 unique, natively validated
   embodiments for each canonical design space.
2. Train a Transformer VAE for exactly 200 epochs on each corpus.
3. Sample posterior latent codes and freeze a six-cluster K-means partition.
4. Under the matched online wall-clock budget, train six independent
   morphology-conditioned Transformer PPO controllers and run LOKI's
   cluster-local candidate replacement.

The first three stages are preparation and are not charged to the online
controller budget. Their durations and artifact hashes are recorded separately.
The standard launchers never generate these artifacts implicitly.

#### Prepare every LOKI artifact

The corpus launcher covers 71 supported presets through 36 canonical corpora.
`TruckUnload-Multi` is intentionally deferred.
Presets with the same embodiment representation, sampling distribution, and
validity rules share one corpus. Collection stores only unique designs that pass
`validate_embodiment`; the first accepted design and every 10,000th accepted
design also undergo a native reset audit.

```bash
# Stage 1: generate or resume all canonical corpora.
python -m loki.collect_all \
  --output-dir data/loki \
  --count 500000

# Stage 2: train one VAE per GPU, dynamically taking the next queued corpus.
python -m loki.train_all_vaes --gpus 0 1 2 3 4 5 6 7

# Stage 3: extract posterior latents on the GPUs and fit six clusters per corpus.
python -m loki.cluster_all --gpus 0 1 2 3 4 5 6 7
```

All three commands accept a subset (`--presets` for collection and `--corpora`
for VAE training or clustering) and `--dry-run`. Collection resumes any corpus
with an existing compatible manifest. VAE training resumes an existing
`last.pt` automatically and skips a completed fixed-recipe run. Clustering
skips an existing compatible artifact and rejects stale or incompatible output.
Use `--skip-unavailable` in the two GPU launchers when some preceding artifacts
are still being prepared.

`collect_all` runs canonical corpora sequentially and records
`data/loki/collection_summary.json`. Racing and Warehouse use JAX on CUDA
ordinal zero; set `CUDA_VISIBLE_DEVICES` on the master process to select the
physical GPU. `train_all_vaes` and `cluster_all` run one process on every ID in
`--gpus`, schedule the estimated longest jobs first, and write per-corpus logs
and suite summaries under `artifacts/loki`. The clustering launcher divides CPU
threads among active K-means jobs. Warehouse-Congested uses a conservative batch
size of 1,024 in both GPU launchers; other corpora use 4,096 by default.

The VAE follows the released recipe: independent posterior mean and log-variance
Transformers, four layers, hidden size 32, four heads, feed-forward factor 8,
Adam at `1e-4`, weight decay `1e-5`, batch size 4,096, KL weight `1e-2`, and a
fixed 200 epochs. Design IDs provide a deterministic 90/10 split. Validation
samples the posterior every ten epochs, and the best reconstruction checkpoint
is retained. K-means consumes one sampled, flattened per-token posterior code
for every design. Corpus, VAE, and clustering manifests bind downstream output
to exact upstream hashes.

Codesign embodiments use a lossless typed token schema rather than fabricated
limbs: scalar parameters become scalar tokens, native rows remain rows,
NeroGrasp uses one XYZ token per mutable vertex, and binary or categorical
fields use matching reconstruction heads. NeroGrasp corpus proposals use
independent +/-0.3 mm mutable-vertex perturbations by default. See
[`loki/README.md`](loki/README.md) for every sampling distribution, corpus group,
artifact, resume rule, and single-stage command, and
[`loki/UPSTREAM.md`](loki/UPSTREAM.md) for exact upstream provenance and porting
choices.

#### Run online LOKI

After the canonical corpus and cluster artifact exist, run one seed directly:

```bash
python -m loki.gym_train \
  --env-id BallCatcher-Pitch \
  --hours 3 \
  --num-envs 16384 \
  --evaluation-worlds 128 \
  --rollout-steps 19 \
  --corpus data/loki/BallCatcher-Pitch \
  --cluster-artifact artifacts/loki/BallCatcher-Pitch/clusters \
  --device cuda:0 \
  --seed 0 \
  --output-dir exp_results/ballcatcher-pitch-loki/loki_seed_0/artifacts \
  --project cdgym-ball-catcher
```

The run splits `--num-envs` as evenly as possible across six controller
runtimes. Each runtime owns an actor Transformer, critic Transformer, PPO
optimizer, observation normalizer, 20-embodiment elite pool, and independent
random-number streams. The actor and critic each use five layers, hidden size
128, two heads, feed-forward width 1,024, learned positions, no dropout, and a
fixed continuous-action standard deviation of 0.9. A controller sees one global
task-state token plus the installed embodiment's typed tokens, and trains over
heterogeneous embodiments from its own frozen cluster.

Every two synchronized PPO updates, each cluster scores 128 randomly sampled
within-cluster candidates for 200 steps and replaces its two worst elite slots
with the two highest-scoring valid candidates. This is literal LOKI replacement:
there is no improvement threshold. The coordinator commits replacements only
after all six scoring rounds complete, freshly scores the elite pools, then
ranks the clusters. The best cluster supplies one controller and its top N
distinct embodiments for the N evaluation worlds.

The wall-clock budget begins at process launch and covers loading,
initialization, PPO, candidate scoring, ranking, checkpointing, and evaluation
scheduling. Evaluation runs independently with deterministic actions at 5%,
10%, ..., 95% of the budget and never changes training state. Checkpoints retain
all six policies, optimizers, normalizers, elite pools, RNG states, counters,
selection state, and artifact identities. W&B receives run and per-cluster
reward, throughput, replacement, rejection, timing, memory, selection, and
evaluation metrics when `--project` is supplied.

The literal single-cluster CLI retains upstream's 2,560 rollout steps. The
comparison launchers default to 19 steps per world because 16,384 total worlds
become about 2,731 worlds per cluster, preserving roughly 51,200 transitions per
cluster update while producing frequent updates. Change it with
`--loki-rollout-steps` in a master launcher.

## Master training scripts

The master launchers create a fixed queue of method/seed jobs and keep one child
process active per requested physical GPU. Each child sees only its assigned GPU
through `CUDA_VISIBLE_DEVICES` and therefore receives `--device cuda:0`.
Completed jobs free their GPU for the next queued job. Each run gets a dedicated
directory and `train.log`; the root `manifest.json` records configuration,
commands, assigned GPUs, status, return codes, elapsed time, and sampled peak
VRAM when a limit is configured.

Output directories must not already exist. Use `--dry-run` first to print the
exact child commands without creating them. `--hours` is the wall-clock budget
inside each trainer; the launcher adds `--shutdown-grace` before its watchdog
terminates a child that did not exit. `--vram-limit-gb` adds a sampled GPU-memory
watchdog and should be used only when each listed GPU is dedicated to the run.

### One environment across methods and seeds

[`launch_comparison.py`](launch_comparison.py) supports `fasttd3`, `ppo_ngopt`,
`ppo_llm`, `cmaes`, `loki`, and `fasttd3_cmaes`. The following launches five LOKI seeds with one
seed on each GPU:

```bash
python launch_comparison.py \
  --env-id BallCatcher-Pitch \
  --methods loki \
  --hours 3 \
  --num-envs 16384 \
  --evaluation-worlds 128 \
  --seeds 0 1 2 3 4 \
  --gpus 1 2 3 4 5 \
  --output-dir exp_results/ballcatcher-pitch-loki
```

Jobs are queued in seed-major, method-minor order. If there are more jobs than
GPUs, each GPU continues taking jobs until the queue is empty. LOKI resolves a
preset to its canonical corpus under `--loki-corpus-root` (default `data/loki`)
and cluster artifact under `--loki-artifact-root` (default `artifacts/loki`).
The default method set is FastTD3, PPO + NGOpt, and CMA-ES, so pass
`--methods loki` or include `loki` explicitly when it is required.

### Environment-family suite

[`launch_environment_suite.py`](launch_environment_suite.py) expands the same
job model across environment families. It supports the BallCatcher, NeroGrasp,
LQ, Microgrid, Network, Racing, RotateInHand, HandPickUp, SoftWalker,
SolarCleaner, and Warehouse families, and the same six methods as
`launch_comparison.py`. RotateInHand and HandPickUp run on the mujoco_warp fork
described under Installation.

```bash
python launch_environment_suite.py \
  --families ball_catcher racing warehouse \
  --methods fasttd3 ppo_ngopt cmaes loki \
  --seeds 0 1 2 3 4 \
  --gpus 0 1 2 3 4 5 6 7 \
  --hours 3 \
  --num-envs 16384 \
  --evaluation-worlds 128 \
  --output-dir exp_results/environment-suite
```

The `soft_walker` family is `SoftWalker-Beam3D` alone; the other SoftWalker
presets are in `DEFERRED_ENV_IDS`. Codesign Gym steps SoftWalker worlds one at a
time (one simulator per world, host NumPy results), so launch it separately with
a small `--num-envs`. Held-out evaluation defaults to 8 worlds for SoftWalker
(128 elsewhere), LOKI scores 16 candidates per cluster and search (LOKI's 128
elsewhere), invalid designs score -10, and every proposed design is repaired
into a valid body by [`common/soft_walker_design.py`](common/soft_walker_design.py).
One L4 steps about 7 SoftWalker worlds per second in total however many jobs
share it, so run one SoftWalker job per GPU. LOKI's PPO batch (5120) must fit in
one rollout, so raise `--loki-rollout-steps` to at least 5120 divided by the
smallest cluster's worlds (`--num-envs` // 6; 2560 at 16 worlds):

```bash
python launch_environment_suite.py \
  --families soft_walker \
  --methods loki \
  --seeds 0 1 \
  --gpus 6 7 \
  --hours 3 \
  --num-envs 16 \
  --loki-rollout-steps 2560 \
  --output-dir exp_results/soft-walker-loki
```

Use `--env-ids ID ...` to override family expansion with exact presets. Output
is nested as `ENV_SLUG/METHOD_seed_SEED/`, with one suite-level manifest. As in
the single-environment launcher, LOKI preparation must already be complete.

### Pokemon suite

[`launch_pokemon_comparison.py`](launch_pokemon_comparison.py) is the separate
master launcher for Pokemon's sharded simulator workers. It currently supports
FastTD3, PPO + NGOpt, and CMA-ES; LOKI online training is not registered in this
launcher. It exposes `--num-envs`, `--evaluation-worlds`, `--envs-per-worker`,
and `--shards`, and otherwise follows the same one-job-per-GPU queue and manifest
layout.

```bash
python launch_pokemon_comparison.py \
  --env-id Pokenv-Random \
  --methods fasttd3 ppo_ngopt cmaes \
  --seeds 0 1 2 3 4 \
  --gpus 0 1 2 3 4 \
  --hours 3 \
  --num-envs 8 \
  --evaluation-worlds 8 \
  --envs-per-worker 8 \
  --shards 4 \
  --output-dir exp_results/pokemon-random
```

## Plot local comparison results

Generate seed-level traces, aggregate means, 95% confidence bands, and run-completion counts directly from a launcher output directory:

```bash
python plot_comparison.py exp_results/complete_suite \
  --env-ids NeroGrasp-Tiny Racing-Levine
```

Omit `--env-ids` to plot every preset in the manifest. Each preset produces a six-panel dashboard under `DIRECTORY/plots` in both PNG and SVG format, covering evaluation reward and success rate, training reward, embodiment rejection rate, evaluation divergence rate, and training throughput. Use `--output-dir` to select another destination. Available data from failed or incomplete runs is retained and identified in the legend.

## Development

The root requirements include the test and formatting tools used across the
repository. Run the available test suites from the repository root:

```bash
pytest cmaes_evolutionary/tests ppo_ngopt_rlgo/tests fasttd3_cmaes/tests
```

Weights & Biases logging is optional at runtime. Use `wandb login` for tracked
runs or set `WANDB_MODE=offline`.


### Independent evaluation in the Gym suite

FastTD3, FastTD3 + CMA-ES, PPO + NGOpt (and PPO + LLM), CMA-ES, and LOKI evaluate in a separate
spawned process. The number of training worlds and the optimizer settings are unchanged.
The evaluator uses the run's device and its own held-out environment batch;
training and evaluation can therefore share GPU resources.

A timer captures the latest immutable policy/design snapshot at each 5% boundary
from 5% through 95% of the wall-clock budget (12, 24, ..., 228 minutes for a four
hour run). Evaluation must finish within that same budget; the trainer saves its
final checkpoint at 100%. Snapshots are published by the training thread at safe
points, at most every 30 seconds during regular updates, and immediately when a
new search incumbent is selected. Long simulator steps or search rounds can
leave the latest snapshot unchanged across several evaluation boundaries.
CMA-ES evaluates its initial mean design/controller until a completed candidate
is available, explicitly labeled `eval/snapshot_kind=initial_candidate`.

Results and best/worst videos reach the existing W&B run from a background log
consumer, without waiting for training to return to an evaluation hook. Metrics
include the requested boundary, snapshot version/time/age, actual start and start
delay, rollout duration, render duration, and completion status. If initialization
has not produced a snapshot, the boundary records `no_snapshot`. If evaluation
itself is slow, boundary snapshots queue in order; their start delays are visible,
and unfinished work is marked when the budget ends. Partial episodes never
produce evaluation scores. A completed score can be logged without videos if
there is insufficient time to finish rendering (`eval/video_complete=false`).

Best checkpoints are saved from the evaluated snapshot, not from newer training
weights. FastTD3's `best_checkpoint.pt` is a portable inference snapshot accepted
by `fast_td3.codesign.evaluation`; its final `checkpoint.pt` retains the full
training state. PPO keeps its loadable `.zip` checkpoint plus best embodiment;
CMA-ES keeps its design/controller `.npz` checkpoint. Network-Urban rendering draws
only the requested world instead of rendering the whole vector batch.

### Warehouse and large simulator batches

FastTD3 replay is capped at **8 GiB** by default, and at half the free CUDA memory
measured after environment/model initialization. The requested transition count is
an upper bound. Use `--replay-memory-gb` and `--replay-transitions` to configure it;
effective capacity and bytes are logged. Simulator world count is unchanged.

The Gym CMA trainer defaults to **diagonal CMA-ES**, with 128 candidates per
generation (or the number of available design slots if smaller). All simulator
worlds are used: extra slots repeat candidates, and complete episode returns are
averaged before telling CMA. The standalone continuous CMA interface retains its
existing full covariance default. In the suite launcher, `--cma-population-size`
and `--cma-covariance diagonal|full` select these settings; the trainer uses
`--population-size` and `--covariance`.

Binary designs use unbounded Gaussian logits decoded at zero, with initial
mutation probabilities centered on the reference occupancy rate. Snapshots and
fitness use the decoded designs, not latent logits. Continuous designs retain
their bounded parameterization.

Evaluation scores and matching best checkpoints are published **before video
rendering**. Videos are logged separately without duplicating scores. Warehouse
renders only requested worlds; repeated best/worst indices share rendered frames.
Video rollout work has a 60-second budget (also bounded by the run deadline), so
an incomplete video does not discard a complete score. This limit is cooperative:
a single renderer or encoder call must return before cancellation is observed.

### Embodiment-conditioned Gym controllers

The Gym environment's observation remains unchanged. At the baseline boundary,
PPO + NGOpt/LLM, CMA-ES and FastTD3 + CMA-ES concatenate the installed embodiment, normalized to
[-1, 1], to each controller observation. FastTD3 already includes the selected
embodiment in its actor and critic state. PPO's augmented observation space,
terminal observations, candidate scoring, and held-out evaluation use the same
input. CMA-ES sizes its controller for the augmented input and uses it during
training and evaluation. New snapshots record `embodiment_conditioned=true`; the
evaluator can still read older unconditioned snapshots.
