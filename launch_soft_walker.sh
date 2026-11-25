#!/usr/bin/env bash
# SoftWalker-Beam3D baselines on the 8xL4 box, in one tmux session:
#   window "main": FastTD3, PPO-NGOpt, CMA-ES x 5 seeds, 4 h each, 16 worlds, GPUs 0-6 (one SoftWalker job per GPU).
#   window "loki": LOKI prep on GPU 7 (corpus -> VAE -> clusters), then LOKI x 5 seeds, 3 h each, on whichever
#                  GPUs are free (--wait-for-free-gpus).
# Run from an activated conda env: bash launch_soft_walker.sh   (then: tmux attach -t soft-walker)
set -euo pipefail
cd "$(dirname "$0")"

SESSION=${SESSION:-soft-walker}
PY=$(command -v python)
MAIN_OUT=${MAIN_OUT:-exp_results/soft-walker}
LOKI_OUT=${LOKI_OUT:-exp_results/soft-walker-loki}
LOG_DIR=${LOG_DIR:-exp_results/logs}
# SoftWalker steps each world on its own CUDA stream; 32 hardware work queues (default 8) measured 9.3 -> 13 world-steps/s
# per L4 at 16 worlds.
CUDA_CONNECTIONS=${CUDA_CONNECTIONS:-32}

if tmux has-session -t "$SESSION" 2>/dev/null; then echo "tmux session $SESSION already exists" >&2; exit 1; fi
for dir in "$MAIN_OUT" "$LOKI_OUT"; do
    if [ -e "$dir" ]; then echo "$dir already exists; the launcher refuses to reuse an output dir" >&2; exit 1; fi
done
mkdir -p "$LOG_DIR"

# Each window runs its own job script (typing long commands into a shell with send-keys is unreliable) and keeps a
# shell open afterwards so the output stays inspectable.
cat > "$LOG_DIR/soft_walker_main.sh" <<EOF
set -o pipefail
cd "$PWD"
export CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_CONNECTIONS
"$PY" launch_environment_suite.py --families soft_walker --methods fasttd3 ppo_ngopt cmaes \\
  --seeds 0 1 2 3 4 --gpus 0 1 2 3 4 5 6 --hours 4 --num-envs 16 --output-dir "$MAIN_OUT" 2>&1 | tee "$LOG_DIR/soft_walker_main.log"
EOF
cat > "$LOG_DIR/soft_walker_loki.sh" <<EOF
set -eo pipefail
cd "$PWD"
export CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_CONNECTIONS
{
  # collect_all only takes --device cpu|cuda|cuda:0; pick the physical GPU with CUDA_VISIBLE_DEVICES.
  CUDA_VISIBLE_DEVICES=7 "$PY" -m loki.collect_all --presets SoftWalker-Beam3D --output-dir data/loki --count 500000 --device cuda:0
  "$PY" -m loki.train_all_vaes --corpora SoftWalker-Beam3D --gpus 7
  "$PY" -m loki.cluster_all --corpora SoftWalker-Beam3D --gpus 7
  "$PY" launch_environment_suite.py --families soft_walker --methods loki --seeds 0 1 2 3 4 \\
    --gpus 0 1 2 3 4 5 6 7 --wait-for-free-gpus --hours 3 --num-envs 16 --loki-rollout-steps 2560 --output-dir "$LOKI_OUT"
} 2>&1 | tee "$LOG_DIR/soft_walker_loki.log"
EOF

tmux new-session -d -s "$SESSION" -n main -c "$PWD" "bash '$LOG_DIR/soft_walker_main.sh'; echo '[main finished]'; exec bash"
tmux new-window -t "$SESSION" -n loki -c "$PWD" "bash '$LOG_DIR/soft_walker_loki.sh'; echo '[loki finished]'; exec bash"
echo "Started tmux session $SESSION (windows: main, loki). Attach with: tmux attach -t $SESSION"
echo "Logs: $LOG_DIR/soft_walker_main.log, $LOG_DIR/soft_walker_loki.log"
