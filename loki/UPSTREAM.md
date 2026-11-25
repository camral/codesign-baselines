# LOKI upstream attribution

The Transformer VAE architecture and training recipe in this directory are
adapted from **Convergent Functions, Divergent Forms** and its released LOKI
implementation:

- Repository: <https://github.com/yeonsumia/loki>
- Pinned reference commit: 24aaac7eb2ecd8f1aada1e1ed0e2d91577fb578a
- Referenced files: vae/model.py, vae/train.py, vae/latent_cluster.py, metamorph/metamorph/algos/ppo/ppo_loki.py, metamorph/metamorph/algos/ppo/model.py, metamorph/metamorph/algos/ppo/transformer.py, metamorph/metamorph/utils/meter.py, metamorph/metamorph/utils/optimizer.py, and scripts/train_loki.sh
- Upstream copyright: Copyright (c) 2025 Hyeonseong Jeon
- Upstream license: MIT

The MIT permission and warranty notice follows.

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Codesign Gym adaptations

The original input schema represents variable UNIMAL limb trees with depth and
EOS features. Codesign Gym corpora instead use their native fixed schema:

- Continuous scalar parameters become one token each.
- Row-structured parameters retain their native row grouping.
- NeroGrasp uses one XYZ token per mutable mesh vertex.
- Native categorical and binary values use categorical cross entropy and binary
  cross entropy heads.
- Fixed token sequences use learned positions and type embeddings; no artificial
  limb depth, adjacency, EOS, or padding tokens are introduced.
- Validation, training, and the later LOKI clustering stage use posterior
  samples.

## Recipe provenance

The following settings come directly from the released `scripts/train_vae.sh`
launch recipe: four Transformer layers, hidden size 32, depth embedding size
32, four attention heads, feed-forward factor 8, Adam learning rate `1e-4`,
weight decay `1e-5`, initial KL weight `1e-2`, minimum KL weight `1e-5`, and
200 epochs. The upstream Python parser defaults to batch size 128, but the
released shell launcher overrides it with batch size 4096.

This port keeps batch size 4096, the independent posterior mean/log-variance
Transformer stacks, per-token latent sampling, causal Transformer decoder,
reconstruction objective, scheduler factor `0.95` with patience 10, and KL
decay factor `0.7` with patience 10. Validation follows the upstream epoch
cadence: epochs 0, 10, 20, and so on. The run uses the released fixed
200-epoch schedule and strict validation improvement rule. The following
choices are Codesign adaptations:

- The KL weight is clamped at its configured minimum.
- The deterministic design-ID split implements the same 90/10 intent for the
  local sharded corpus format.

## Clustering provenance

The implementation in `cluster.py` directly preserves the released
`vae/latent_cluster.py` choices that define cluster membership: draw one sample
from each per-token posterior, flatten the `[token, hidden]` latent array, and
fit `sklearn.cluster.KMeans(n_clusters=..., random_state=...)` using its normal
defaults. The Codesign port uses the agreed six clusters instead of upstream's
CLI default of ten.

The local code streams sharded numeric corpora into a disk-backed NumPy array
instead of concatenating every GPU tensor, records separate posterior and
K-means seeds, verifies corpus/checkpoint hashes, and saves immutable labels plus
per-cluster design-ID indexes. It does not decode and revalidate VAE
reconstructions before clustering because every stored Codesign corpus row has
already passed its native validator and cluster ownership must cover each stored
design exactly once. Source corpus files are never moved into cluster folders.

## Online controller and search provenance

The one-cluster path preserves the released 20-walker configuration used by
the paper artifact scripts, search every two completed PPO iterations, 128
random within-cluster candidates, two replacements, no improvement threshold,
stochastic policy actions, a 200-step scoring window, and rolling ten-episode
elite returns. It also preserves the controller dimensions, independent actor
and critic Transformers, fixed continuous-action standard deviation, PPO
rollout and optimizer settings, cosine schedule, and warmup.

Codesign tasks do not expose UNIMAL limb observations or per-limb actuator
slots. The port therefore feeds one global task-state token and the corpus's
typed embodiment tokens to each Transformer, then predicts the native task
action from the global token. Native categorical task actions use SB3's
categorical policy distribution; continuous actions use the released fixed
standard deviation of 0.9. Observation normalization uses a separate saved
state per controller.

Upstream computes the median of completed episode returns during candidate
scoring. Some Codesign episodes do not complete in 200 steps, which would make
that median undefined. This port uses the same median whenever at least one
episode completes and otherwise uses the 200-step partial return. For presets
with several physical worlds per logical embodiment, it averages their scores.
Any reset rejection excludes the logical candidate rather than scoring its
fallback embodiment.
