# Generate the 36 canonical corpora

From the repository root, with the codesign Python environment active:

```bash
python -m loki.collect_all --output-dir data/loki --count 500000
```

This maps all 71 supported presets onto 36 reusable corpus groups and runs
one canonical preset per group sequentially, in separate processes, using one
visible GPU by default. Racing and Warehouse explicitly use the CUDA backend
and expose only CUDA ordinal zero to JAX; `--device cpu` instead forces JAX CPU. Use `CUDA_VISIBLE_DEVICES=2` before the command to select physical GPU 2. A preset's native simulator dependencies must be installed.
The launcher discovers a sibling `pokemon-codesign-env` source checkout when
present; use `--pokemon-source /path/to/pokemon-codesign-env` to select another.

The shared groups are BallCatcher (3 presets), Microgrid at the 300 kW and
1200 kW scales (3 and 2), Racing (2), SolarCleaner (3), HandPickUp with RotateInHand (4), NeroGrasp (13), each
Pokémon league generation pair, and the four Generation 9 league variants.
The six MuJoCo locomotion presets and three custom-game Pokémon presets remain separate. LQStructure, Network,
SoftWalker-Beam3D, TruckUnload-Single, and Warehouse presets also remain separate. `CORPUS_GROUPS` in
[`collect_all.py`](collect_all.py) is the executable mapping.

Each corpus is stored under its canonical preset ID in `data/loki/PRESET`. The terminal and `collection_summary.json` list every preset served by each
corpus. Existing manifests automatically enable `--resume`, including completed corpora. The collector checks identity
and schema compatibility; keep the same seed, device, overrides, and audit cadence
when resuming. A failed identity check never overwrites the existing corpus.
If an earlier standalone run used a different directory, that corpus must remain
at `OUTPUT_DIR/PRESET` to be discovered by this launcher.

Output goes to `data/loki/logs/PRESET.log` (appended across launches). The terminal
prints starts and completions. `data/loki/collection_summary.json` records the
latest invocation's results and is updated after each corpus. Failed or stalled
corpora do not prevent later corpora from running; the launcher exits with code 1
if any failed. Ctrl-C stops the launch sequence and lets the current collector
flush its progress, with a 30-second shutdown timeout. Rerun the same command to
resume. Run only one master launcher per output directory at a time.

Preview commands without creating corpora:

```bash
python -m loki.collect_all --dry-run
```

Pilot all presets in a separate directory before the full run:

```bash
python -m loki.collect_all --output-dir data/loki_pilot --count 10 --audit-every 1
```

Select a subset or retry particular presets. Selecting multiple members of the same group launches its canonical corpus once:

```bash
python -m loki.collect_all --presets BallCatcher-Arc BallCatcher-Fielding Pokenv-Random
```

The launcher also accepts `--seed`, `--device`, `--shard-size`, `--audit-every`,
`--max-attempts`, and `--max-stalled-attempts`. An optional `--overrides FILE`
accepts JSON mapping preset IDs to native environment kwargs. An override supplied under any group member applies to its canonical corpus;
conflicting overrides within one group are rejected. For example:

```json
{"NeroGrasp-Tiny": {"sample_perturbation_m": 0.0002}}
```

The target is 500,000 **unique validated** designs per canonical corpus, or
18 million across all 36 corpora. Reaching that target depends on the valid-design count and acceptance rate of
each space. An incomplete corpus is reported as a failure rather
than padded with duplicates. See [collector documentation](README.md) for sampling
distributions, audit semantics, and corpus format.
