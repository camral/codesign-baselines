# PPO + LLM

Ollama Cloud `nemotron-3-super:cloud` replaces NGOpt proposals by default.
The entry point calls the shared PPO+NGOpt runner: one retained PPO iteration
(256 steps per world, unchanged PPO hyperparameters), then a population of perturbed
LLM templates, scored for one episode with the current deterministic PPO policy.
PPO trains on the GPU while template inference uses the remote Ollama API.
The LLM is not fine-tuned and does not generate controller actions or executable code.

Each prompt contains the task, ordered parameter bounds, base embodiment, and
the complete table of template statistics from all completed rounds. Override
`--llm-task-description` for more task context. BallCatcher includes arm body
names when available. Other environments use their flattened parameter paths.
This supports the same finite numeric embodiment spaces as the PPO+NGOpt adapter,
not arbitrary mesh topology. All templates in a round use the same feedback;
Ollama requests four templates per API call and assembles all 16 before evaluating a round; uniqueness is not guaranteed.

The default is 16 templates: at 16,384 worlds each gets 1,024 independently
perturbed copies. `--llm-templates` must divide the population size; for small
smoke tests it is capped at the population size. Template assignments are shuffled
with a separate seeded generator. Gaussian noise has standard deviation
`--llm-perturbation-std` (default 0.05) times each parameter's full range. Copies
are clipped to bounds; integer parameters are rounded. A value of zero permits
an unperturbed ablation. No MeshSpace.sample() calls are used.

Feedback reports mean, population standard deviation, minimum and maximum reward,
sample count and rejection rate for each template in each round. The mean includes
rejection penalties and measures the perturbed neighborhood, not the exact template.
Round labels preserve the distinction between scores from changing PPO policies.
BallCatcher context is read from the live environment: launch mode and distribution,
reward/dwell/drop thresholds, episode limit, body names and physical offset scales.
Search state includes remaining seconds, completed PPO iterations, population sizes,
and the best observed design/reward. Feedback also includes goal-entry, dwell-catch,
post-dwell-drop, grounded-success, ever-grounded, ball-left-field, time-limit and
divergence rates when the environment exposes them, plus mean episode length.
These count events only during the first evaluated episode; rejected copies have
zero events and length and remain in the denominator. Unsupported signals are omitted.
For non-BallCatcher tasks, parameter schema and rejection penalty are supplied;
use `--llm-task-description` for task-specific semantics.
The shared runner still selects its best actual evaluated design for held-out
evaluation, and uses the actual perturbed population for the next PPO iteration.
Complete history is retained in local artifacts. Prompt overflow normally raises an explicit error without silently dropping rows; Gemma is the exception described below. Local Qwen also needs sufficient VRAM for larger histories.

Strict JSON parsing enforces dimensions, finite values, bounds and integer fields.
Malformed complete proposal batches are retried until valid; no random fallback is used. Environment-level rejection receives the same rejection score
and subsequent training fallback as PPO+NGOpt. A sampling round that starts before the comparison deadline may finish after any number of retriable provider failures and receives one fixed 120-second charge. Runtime/model/OOM and non-retriable provider errors fail visibly.

## Install and launch

From `codesign-baselines`, using its installed `codesign` environment:

```bash
conda activate codesign
export OLLAMA_API_KEY=your_ollama_api_key
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
python launch_comparison.py --methods ppo_llm --env-id BallCatcher-Pitch \
    --seeds 0 --gpus 4 --hours 4 --num-envs 16384 \
    --project cdgym-ball-catcher --entity projectavi --group ppo_llm_pitch \
    --output-dir exp_results/ppo_llm_pitch
```

Choose an available training GPU. EGL device 0 is the rendering device index,
not the CUDA-visible training index. Both launchers accept `--methods ppo_llm`;
existing default methods are unchanged. Collate with `python collate_comparison.py
exp_results/ppo_llm_pitch`.


For hosted Gemma 4 31B, set the Google API key and select the Gemma gateway. The default model is `gemma-4-31b-it`, with a 250000-token prompt guard and up to 32768 output tokens. It requests high thinking and validates JSON locally; the hosted Gemma API does not document provider-enforced response schemas.

```bash
export GEMINI_API_KEY=your_google_api_key
python launch_comparison.py --methods ppo_llm --llm-gateway gemma \
    --env-id BallCatcher-Pitch --seeds 0 --gpus 4 --hours 4 --num-envs 16384 \
    --project cdgym-ball-catcher --entity projectavi --group ppo_llm_gemma_pitch \
    --output-dir exp_results/ppo_llm_gemma_pitch
```

Remote gateways accept only complete requested chunks. Ollama assembles four four-template chunks; Gemma requests all 16 templates at once. Gemma accepts plain JSON or a single JSON code fence, then checks the exact template count before local shape and bounds validation. Before each generation, the gateway calls Google's countTokens endpoint and compares the exact prompt count with the --llm-gemma-input-tpm default of 16000. If necessary, it binary-searches the minimum oldest-first template-row truncation that fits, while retaining the full table in local artifacts and logging total and dropped row counts. It then paces every generation attempt, including retries; other project traffic is not visible to the gateway. Actual provider latency and retries are excluded from the comparison budget, and every sampling round is charged exactly 120 seconds instead. The comparison launcher disables its real-time deadline watchdog for PPO+LLM and lets this modeled trainer budget determine completion. Actual generation latency remains logged as `llm/generation_seconds`, while the charged time is logged as `llm/modeled_sampling_seconds`. API keys are read from environment variables and are never included in run configuration or artifacts.

Before live Gemma training, rerun `python -m ppo_llm_rlgo.gemma_batch_probe --mode one-by-sixteen --save-full-response` with an audited `--prompt-file` and fresh `--output`. The result JSONL stores the full model completion but not the API key or prompt; require `http_status=200` and `valid_designs=16` before using that output format in training.

## NVIDIA NIM

Set `NVIDIA_API_KEY` and choose one of three hosted NIM aliases: `nvidia-kimi` selects `moonshotai/kimi-k3`, `nvidia-deepseek` selects `deepseek-ai/deepseek-v4-flash-0731`, and `nvidia-nemo` selects `nvidia/nemotron-3.5-lightning-30b-a3b`. Each alias assembles each 16-template round from four sequential four-template chat completions, accepts plain or singly fenced JSON, and uses the documented maximum reasoning configuration: Kimi `reasoning_effort=max`, DeepSeek thinking with `reasoning_effort=high`, and Nemotron thinking with a 16384-token reasoning budget. All use the NIM-required temperature 1.0 and top-p 0.95, default to a 1M context guard and 16384 output tokens, and use --llm-api-timeout=900 seconds per response read. Kimi consumes its documented server-sent event stream incrementally to avoid waiting for the complete reasoning response before receiving bytes; Nemotron uses its working JSON response path. Each sampling round receives the same fixed 120-second comparison-budget charge regardless of provider latency or retries.

```bash
export NVIDIA_API_KEY=your_nvidia_api_key
python launch_comparison.py --methods ppo_llm --llm-gateway nvidia-kimi --env-id BallCatcher-Pitch --seeds 0 --gpus 0 --hours 4 --num-envs 16384 --project cdgym-ball-catcher --entity projectavi --group ppo_llm_nvidia_kimi_pitch --output-dir exp_results/ppo_llm_nvidia_kimi_pitch
```

Replace `nvidia-kimi` in that command with `nvidia-deepseek` or `nvidia-nemo` and use a fresh group and output directory for each model. The implementation calls NVIDIA directly with the standard library; it does not require the OpenAI package.

## Manual LLM loop

To act as the embodiment sampler yourself, run the interactive entry point in a
terminal. It fixes simulation and EGL rendering to GPU 0, performs one PPO
iteration, prints and saves a standalone prompt, and waits without consuming the
training budget:

```bash
conda activate codesign
python -m ppo_llm_rlgo.manual_train --env-id BallCatcher-Pitch --hours 4 \
    --output-dir exp_results/manual_pitch
```

Copy the active prompt from `current_prompt.txt` into any LLM. The same exact text
is archived as `manual_prompts/round_NNNN.txt` for reproducibility. If it accepts
images, attach the `manual_renders/round_NNNN/contact_sheet.png` path named in the
prompt. Paste its complete response back into the training terminal as one object:

```json
{"embodiments": [[45 values], [45 values], [45 values], [45 values], [45 values], [45 values], [45 values], [45 values], [45 values], [45 values], [45 values], [45 values], [45 values], [45 values], [45 values], [45 values]]}
```

The dimension is environment-derived; Pitch currently reports 45 parameters, and
the prompt contains every index, bound, body and local-axis direction. Multiline
JSON is accepted automatically once complete. Invalid input rejects the entire
batch and presents the same round again; enter `END` to submit an otherwise
incomplete response, or `/quit` to save the current policy and exit cleanly. Each
valid template receives an equal number of randomly perturbed evaluation worlds.
After scoring, the next PPO iteration trains on those perturbed embodiments. The
next boundary renders all 16 exact templates in labeled individual PNGs and a
contact sheet, then includes their aggregate reward and diagnostic statistics in
the next portable prompt.

Ollama defaults: four requests of four templates for each 16-template round, approximate context
guard 250000 tokens, max 32768 output tokens, full thinking, temperature 0.8 and
top-p 0.8. `--llm-model` and `--llm-task-description` are configurable. No local
Ollama daemon or model download is needed for direct API access. Initialization, simulation, PPO and evaluation use measured time. LLM inference and retries use the fixed 120-second per-round charge.
Evaluation runs at the shared 5% boundaries (checked between sampling rounds):
current PPO plus best scored embodiment in every evaluation world, with best/worst
videos, reward logging and best-evaluation checkpoint selection unchanged.

## Memory and reproducibility

Ollama Cloud does not place model weights or KV-cache on the training GPU. Provider requests and exponential retry backoff do not advance the comparison budget. HTTP 429, 500, 502, 503 and 504 responses retry until the sampling round succeeds; malformed batches use bounded 1-to-30-second backoff. Attempt, success, HTTP-error and retry counters plus the latest provider error are written to local and W&B metrics. Ollama proposal audit records each response stop reason, generated-token count, and thinking/content lengths without recording the reasoning trace itself. Use --llm-no-think only for an explicit no-thinking ablation; thinking stays enabled by default.
The optional local Qwen 4B BF16 weights alone need roughly 8 GB. Generation microbatches bound KV-cache
and activation memory, but total 40-GB fit with 16K simulation worlds still requires
a native measurement. PyTorch peak memory metrics exclude Warp/driver allocations;
use external GPU memory measurements as well. Lower `--llm-batch-size` if needed.
Only the templates require text generation; the remaining population is generated
by the explicit perturbation rule above. Full history increases generation memory
and time as the run proceeds. History uses a column header plus numeric rows so field names are not repeated. Parameter metadata is losslessly factored into shared defaults and varying rows. Historical embodiments are losslessly represented as index-value changes from the supplied base embodiment. Every template remains in artifacts; only the Gemma request may omit the minimum number of oldest rows needed for its input-rate ceiling. Prompt floats use seven significant digits while artifact files retain Python's full serialized precision. `--llm-batch-token-budget` defaults to 32768 and automatically lowers
the optional Hugging Face generation microbatch as prompts grow. Ollama receives four templates per request, preserving the full 16-template round.

Run artifacts include `llm_config.json` (gateway model and schema),
`llm_proposals.jsonl` (prompts, raw completions and parsing failures),
`llm_population.jsonl` (every perturbed design and its template assignment),
`llm_feedback.jsonl` (all candidate rewards/rejections),
`llm_template_statistics.csv` and `.json` (complete per-round template table), and `llm_sampler.json`
(final feedback/counters/model reference). Standard PPO checkpoints and local/W&B
metrics are shared with PPO+NGOpt. Optional local weights are cached once, not copied
into every checkpoint. Neither runner implements resume. LLM sampling preserves
the PPO CPU/CUDA random streams.

## Model gateway

`gateway.py` defines `LLMGateway`: `count_tokens(messages)`,
`generate(messages, count, budget, seed=...)`, `metrics()`, `close()`, and
`resolved_revision`. The default `--llm-gateway ollama` uses direct Ollama Cloud
API access at `https://ollama.com/api/chat` and `OLLAMA_API_KEY`. Select `--llm-gateway gemma` for hosted Gemma 4 31B (`gemma-4-31b-it`) through the Google `generateContent` endpoint and `GEMINI_API_KEY`. The three `nvidia-*` aliases use NVIDIA's OpenAI-compatible hosted chat-completions endpoint and `NVIDIA_API_KEY`. Ollama Cloud does
not currently enforce structured outputs, so complete JSON batches are checked
locally and retried atomically. API usage comes from response token counts; the
prompt-growth guard is an approximation. For optional local Qwen, install
`ppo_llm_rlgo/requirements.txt`, download the model, and pass
`--llm-gateway huggingface --llm-model Qwen/Qwen3-4B-Instruct-2507
--llm-context-tokens 65536 --llm-max-new-tokens 256`.
To add another gateway, implement this interface and select
`--llm-gateway your_module:factory`; the factory receives the parsed arguments.
It must return one text completion per requested template, respect the gateway timeout configuration and fixed-round budget accounting, and expose its model/version identifier.
Credentials should be read from environment variables, not written into run config.
Tests may inject a gateway
directly via `LLMSampler(..., gateway=...)`.

Validated dependencies: Transformers 4.57.6 and Accelerate 1.15.0. On 2026-09-15,
the native GPU-0 smoke used 32 BallCatcher-Pitch worlds, two templates, two held-out
worlds and local Qwen for 108 seconds. It completed 28 search rounds, generated
58 valid templates with zero malformed outputs, and saved held-out evaluation
videos. The last population was interrupted by the deadline and excluded from
the 56-row feedback table. PyTorch peak allocated/reserved memory was approximately
10.72/19.08 GiB. This small-population test does not establish A100 memory fit at
maximum context: history growth increases KV-cache memory, particularly with
`--llm-batch-size 8`; use a smaller generation batch for long histories on 40GB.

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_ppo_llm.py tests/test_wall_clock.py tests/test_timed_ppo.py tests/test_hyperparameter_audit.py tests/test_launcher.py -q
```
