"""Dependency-light CLI configuration shared with the experiment launchers."""

import math

DEFAULT_MODELS = {"ollama": "nemotron-3-super:cloud", "gemma": "gemma-4-31b-it", "nvidia-kimi": "moonshotai/kimi-k3", "nvidia-deepseek": "deepseek-ai/deepseek-v4-flash-0731", "nvidia-nemo": "nvidia/nemotron-3.5-lightning-30b-a3b", "huggingface": "Qwen/Qwen3-4B-Instruct-2507"}
DEFAULT_CONTEXT_TOKENS = {"ollama": 250000, "gemma": 250000, "nvidia-kimi": 1000000, "nvidia-deepseek": 1000000, "nvidia-nemo": 1000000, "huggingface": 65536}
DEFAULT_MAX_OUTPUT_TOKENS = {"ollama": 32768, "gemma": 32768, "nvidia-kimi": 16384, "nvidia-deepseek": 16384, "nvidia-nemo": 16384, "huggingface": 256}
DEFAULT_MODEL = DEFAULT_MODELS["ollama"]


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("must be finite and positive")
    return value


def resolve_arguments(args):
    if not hasattr(args, "llm_api_timeout"):
        args.llm_api_timeout = 900.
    if not args.llm_model:
        args.llm_model = DEFAULT_MODELS.get(args.llm_gateway, DEFAULT_MODEL)
    if args.llm_context_tokens is None:
        args.llm_context_tokens = DEFAULT_CONTEXT_TOKENS.get(args.llm_gateway, 65536)
    if args.llm_max_new_tokens is None:
        args.llm_max_new_tokens = DEFAULT_MAX_OUTPUT_TOKENS.get(args.llm_gateway, 4096)
    return args


def add_arguments(parser):
    parser.add_argument("--llm-gateway", default="ollama", help="ollama, gemma, nvidia-kimi, nvidia-deepseek, nvidia-nemo, huggingface, manual or module:factory implementing LLMGateway")
    parser.add_argument("--llm-templates", type=int, default=16, help="Template count; must divide num-envs (capped at num-envs)")
    parser.add_argument("--llm-perturbation-std", type=float, default=.05, help="Gaussian standard deviation as a fraction of each parameter range")
    parser.add_argument("--llm-model", default="", help="Provider-specific default when omitted")
    parser.add_argument("--llm-revision", default="main")
    parser.add_argument("--llm-batch-size", type=int, default=8)
    parser.add_argument("--llm-context-tokens", type=int, default=None, help="Provider-specific context limit when omitted")
    parser.add_argument("--llm-batch-token-budget", type=int, default=32768, help="Maximum prompt tokens across a generation microbatch")
    parser.add_argument("--llm-max-new-tokens", type=int, default=None, help="Provider-specific output limit when omitted")
    parser.add_argument("--llm-api-timeout", type=positive_float, default=900., help="Maximum seconds without a response byte for one remote LLM request")
    parser.add_argument("--llm-gemma-input-tpm", type=int, default=16000, help="Conservative Gemma input-token pacing limit per minute")
    parser.add_argument("--llm-temperature", type=float, default=.8)
    parser.add_argument("--llm-top-p", type=float, default=.8)
    parser.add_argument("--llm-think", dest="llm_think", action="store_true", default=True, help="Enable Ollama model thinking (default)")
    parser.add_argument("--llm-no-think", dest="llm_think", action="store_false", help="Disable Ollama thinking as an explicit ablation")
    parser.add_argument("--llm-task-description", default="")
    parser.add_argument("--llm-local-files-only", action="store_true")


def command_arguments(args):
    command = []
    for key, value in vars(args).items():
        if key.startswith("llm_"):
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                command.extend([flag] if value else ["--llm-no-think"] if key == "llm_think" else [])
            elif value is not None:
                command.extend([flag, str(value)])
    return command
