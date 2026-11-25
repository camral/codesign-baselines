"""LLM templates, balanced perturbed populations and complete template feedback."""

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from .config import add_arguments, resolve_arguments
from .gateway import CallableGateway, make_gateway
from .context import task_context


LLM_SAMPLE_ROUND_SECONDS = 120.


def validate_arguments(args):
    resolve_arguments(args)
    if min(args.llm_batch_size, args.llm_context_tokens, args.llm_max_new_tokens, args.llm_templates, args.llm_batch_token_budget) < 1:
        raise ValueError("LLM batch, context, output and template counts must be positive")
    if not math.isfinite(args.llm_api_timeout) or args.llm_api_timeout <= 0:
        raise ValueError("LLM API timeout must be finite and positive")
    if not math.isfinite(args.llm_temperature) or args.llm_temperature <= 0 or not 0 < args.llm_top_p <= 1:
        raise ValueError("LLM temperature must be positive and top-p must be in (0, 1]")
    if not math.isfinite(args.llm_perturbation_std) or args.llm_perturbation_std < 0:
        raise ValueError("LLM perturbation std must be finite and nonnegative")


def parse_design(text, low, high, discrete):
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    payload = json.loads(text)
    if not isinstance(payload, dict) or set(payload) != {"embodiment"}:
        raise ValueError("expected only an embodiment field")
    values = payload["embodiment"]
    if not isinstance(values, list) or any(type(value) not in (int, float) for value in values):
        raise ValueError("embodiment must be a flat numeric list")
    array = np.asarray(values, dtype=np.float64)
    if array.shape != low.shape or not np.isfinite(array).all() or (array < low).any() or (array > high).any():
        raise ValueError("wrong shape, nonfinite value or out-of-bounds embodiment")
    if (array[discrete] != np.rint(array[discrete])).any():
        raise ValueError("discrete parameters must be integral")
    return array.astype(np.float32)


@dataclass
class Candidate:
    value: np.ndarray
    uid: int
    template_id: int


class LLMSampler:
    def __init__(self, env, base, args, output, *, backend=None, gateway=None):
        validate_arguments(args)
        if backend is not None and gateway is not None:
            raise ValueError("supply either backend or gateway")
        self.args, self.output = args, Path(output)
        self.low = np.asarray(env.embodiment_space.low, dtype=np.float64)
        self.high = np.asarray(env.embodiment_space.high, dtype=np.float64)
        if not np.isfinite(self.low).all() or not np.isfinite(self.high).all():
            raise ValueError("PPO+LLM requires finite embodiment bounds")
        from common.codesign_adapter import SpaceCodec
        codec = getattr(env, "design_codec", None)
        codec = SpaceCodec(env.embodiment_space) if codec is None else codec
        self.discrete = np.asarray(codec.discrete, dtype=bool)
        self.base = np.asarray(base).tolist()
        self.parameters = [dict(index=i, low=float(lo), high=float(hi), integer=bool(self.discrete[i])) for i, (lo, hi) in enumerate(zip(self.low, self.high))]
        for leaf in getattr(codec, "leaves", ()):
            for index in range(leaf.start, leaf.stop):
                self.parameters[index]["name"] = ".".join(leaf.path) or "embodiment"
        native = getattr(env, "env", env)
        if hasattr(native, "_segs") and hasattr(native, "m"):
            for body, adr, num in zip(native._segs["body"], native._segs["axis_adr"], native._segs["axis_num"]):
                name = native.m.body(int(body)).name
                for axis in range(adr, adr+num):
                    slot = int(native._segs["axis_act"][axis])
                    self.parameters[slot].setdefault("bodies", []).append(name)
                    self.parameters[slot].setdefault("link_axes", []).append(dict(body=name, direction_local=native._segs["axis_dir"][axis].tolist()))
        self.description = args.llm_task_description or ("Catch the thrown ball with the robot's glove while preserving a stable grounded landing. Embodiment coordinates move leg and arm link ends along their listed local axes; glove dimensions may also change." if args.env_id.startswith("BallCatcher-") else "Optimize embodiment parameters for episodic return under the current PPO controller.")
        self.task = task_context(env, self.parameters)
        self.search_state = {}
        self.diagnostics = {}
        self.history, self.feedback, self.templates = [], [], []
        self.round = self.calls = self.generated = self.invalid = self.invalid_batches = self.accepted = 0
        self.prompt_tokens = 0
        self.generation_seconds = 0.0
        self.modeled_sampling_seconds = 0.0
        self.rng = np.random.default_rng(args.seed)
        self.pending = {}
        self.gateway = gateway if gateway is not None else CallableGateway(backend) if backend is not None else make_gateway(args)
        self.resolved_revision = self.gateway.resolved_revision
        self._write("llm_config.json", dict(model=args.llm_model, revision=self.resolved_revision, gateway=args.llm_gateway, parameters=self.parameters, description=self.description, templates=args.llm_templates, perturbation_std=args.llm_perturbation_std))

    def _write(self, name, value):
        (self.output / name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")

    def prompt(self, template_count=None):
        source_columns = list(dict.fromkeys(key for row in self.history for key in row))
        columns = ["embodiment_changes_from_base_index_value" if column == "embodiment" else column for column in source_columns]
        rows = [[self._history_value(column, row.get(column)) for column in source_columns] for row in self.history]
        content = json.dumps(dict(task=self.args.env_id, description=self.description, task_context=self._compact_task_context(), round=self.round, search_state=self.search_state, base_embodiment=self.base, perturbation_std_fraction=self.args.llm_perturbation_std, template_statistics_columns=columns, template_statistics_rows=rows, history_embodiment_encoding="Each historical embodiment cell contains [parameter_index, value] pairs; omitted indices equal base_embodiment.", diagnostic_definitions="Rates are fractions of evaluated copies with the event at least once during their first episode; rejected copies contribute zero events and zero length. catch_rate requires accumulated dwell and retention; goal_entry_rate includes brief contact with the reward region. Missing signals are omitted."), separators=(",", ":"), allow_nan=False)
        batched = getattr(self.gateway, "batched_templates", False) and template_count is not None
        instruction = f'Propose exactly {template_count} embodiment templates in one response. Return only {{"embodiments": [[numbers in parameter-index order], ...]}} with exactly {template_count} lists and exactly {self.low.size} numeric values in every list.' if batched else f'Propose one embodiment template. Return only {{"embodiment": [exactly {self.low.size} numbers in parameter-index order]}}.'
        messages = [dict(role="system", content=instruction+' Maximize mean episodic return across random perturbations. Respect bounds and integer constraints. Copies receive independent Gaussian noise scaled by parameter range, clipped to bounds and rounded for integers. Use the complete template statistics table to explore and improve robust designs. Statistics measure perturbed copies, not the exact template. Rejected copies receive the rejection penalty and are included in mean_reward. PPO changes each round; compare round labels when interpreting scores. No code or explanation.'), dict(role="user", content=content)]
        tokens = self.gateway.count_tokens(messages)
        self.prompt_tokens = tokens
        required = tokens+self.args.llm_max_new_tokens
        if required > self.args.llm_context_tokens:
            raise ValueError(f"prompt and reserved output require {required} tokens, exceeding --llm-context-tokens={self.args.llm_context_tokens}; no rows were removed")
        return messages

    def _compact_task_context(self):
        context = {key: value for key, value in self.task.items() if key != "parameters"}
        parameters = []
        for source in self.task["parameters"]:
            parameter = dict(source)
            axes = parameter.pop("link_axes", None)
            if axes is not None:
                parameter["directions_local"] = [axis["direction_local"] for axis in axes]
                axis_bodies = [axis.get("body") for axis in axes]
                if axis_bodies != parameter.get("bodies"):
                    parameter["axis_bodies"] = axis_bodies
            parameters.append(parameter)
        columns = list(dict.fromkeys(key for parameter in parameters for key in parameter))
        defaults = {column: parameters[0][column] for column in columns if all(column in parameter and parameter[column] == parameters[0][column] for parameter in parameters)} if parameters else {}
        varying = [column for column in columns if column not in defaults]
        context.update(parameter_defaults=defaults, parameter_columns=varying, parameter_rows=[[parameter.get(column) for column in varying] for parameter in parameters], parameter_encoding="Combine parameter_defaults with each row indexed by parameter_columns; directions_local aligns with bodies unless axis_bodies is supplied.")
        return context

    def _history_value(self, column, value):
        if column == "embodiment":
            value = [[index, item] for index, item in enumerate(value) if item != self.base[index]]
        return self._context_value(value)

    @classmethod
    def _context_value(cls, value):
        if isinstance(value, list):
            return [cls._context_value(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._context_value(item) for key, item in value.items()}
        return float(f"{value:.7g}") if isinstance(value, float) else value

    def ask_batch(self, count, budget, progress=None):
        modeled_elapsed = getattr(budget, "modeled_elapsed", None)
        if modeled_elapsed is None:
            return self._ask_batch(count, budget, progress)
        with modeled_elapsed(LLM_SAMPLE_ROUND_SECONDS):
            result = self._ask_batch(count, budget, progress)
        self.modeled_sampling_seconds += LLM_SAMPLE_ROUND_SECONDS
        return result

    def _ask_batch(self, count, budget, progress=None):
        if self.pending or self.feedback:
            raise RuntimeError("score the pending population before asking for another round")
        template_count = min(self.args.llm_templates, count)
        if count < 1 or count % template_count:
            raise ValueError("llm-templates must divide the candidate/environment count")
        self.templates = []
        self.search_state.update(completed_rounds=self.round, population_size=count, template_count=template_count, copies_per_template=count//template_count, remaining_seconds=getattr(budget, "remaining", None))
        batched = getattr(self.gateway, "batched_templates", False)
        messages = self.prompt(template_count if batched else None)
        malformed_retries = 0
        with (self.output / "llm_proposals.jsonl").open("a", buffering=1) as audit:
            while len(self.templates) < template_count and not budget.expired:
                if progress is not None:
                    progress()
                if budget.expired:
                    break
                remaining_templates = template_count-len(self.templates)
                size = min(getattr(self.gateway, "max_templates_per_request", template_count), remaining_templates) if batched else min(self.args.llm_batch_size, max(1, self.args.llm_batch_token_budget//max(self.prompt_tokens, 1)), remaining_templates)
                if batched:
                    messages = self.prompt(size)
                audit.write(json.dumps(dict(round=self.round, request_templates=size, accepted_templates=len(self.templates), prompt=messages))+"\n")
                started = time.monotonic()
                try:
                    responses = self.gateway.generate(messages, size, budget, seed=self.args.seed+self.calls)
                finally:
                    self.generation_seconds += time.monotonic()-started
                    self.calls += 1
                response_stats = getattr(self.gateway, "last_response", None)
                if response_stats:
                    audit.write(json.dumps(dict(round=self.round, provider_response=response_stats))+"\n")
                if not responses and budget.expired:
                    break
                if len(responses) != size:
                    raise RuntimeError("LLM gateway must return one completion per requested template")
                parsed = []
                for response in responses:
                    self.generated += 1
                    try:
                        value = parse_design(response, self.low, self.high, self.discrete)
                    except (ValueError, TypeError, OverflowError) as error:
                        self.invalid += 1
                        audit.write(json.dumps(dict(round=self.round, response=response, error=str(error)))+"\n")
                        continue
                    parsed.append((response, value))
                if batched and len(parsed) != size:
                    self.invalid_batches += 1
                    delay = min(30., 2.**min(malformed_retries, 5))
                    malformed_retries += 1
                    remaining = getattr(budget, "remaining", None)
                    if remaining is not None:
                        delay = min(delay, max(0., remaining))
                    audit.write(json.dumps(dict(round=self.round, error="entire API template batch rejected; retrying", retry_delay_seconds=delay))+"\n")
                    if delay > 0 and not budget.expired:
                        time.sleep(delay)
                    continue
                malformed_retries = 0
                for response, value in parsed:
                    template_id = len(self.templates)
                    self.templates.append(value)
                    audit.write(json.dumps(dict(round=self.round, template_id=template_id, response=response, embodiment=value.tolist()))+"\n")
        if len(self.templates) != template_count or budget.expired:
            return []
        assignments = np.repeat(np.arange(template_count), count//template_count)
        self.rng.shuffle(assignments)
        values = np.stack(self.templates)[assignments].astype(np.float64)
        values += self.rng.normal(size=values.shape)*self.args.llm_perturbation_std*(self.high-self.low)
        values = np.clip(values, self.low, self.high)
        values[:, self.discrete] = np.clip(np.rint(values[:, self.discrete]), np.ceil(self.low[self.discrete]), np.floor(self.high[self.discrete]))
        candidates = [Candidate(value.astype(np.float32), self.accepted+i, int(template_id)) for i, (value, template_id) in enumerate(zip(values, assignments))]
        self.accepted += count
        self.pending = {candidate.uid: candidate for candidate in candidates}
        with (self.output / "llm_population.jsonl").open("a") as stream:
            for candidate in candidates:
                stream.write(json.dumps(dict(round=self.round, uid=candidate.uid, template_id=candidate.template_id, embodiment=candidate.value.tolist()))+"\n")
        return candidates

    def tell_result(self, candidate, score, rejected):
        if candidate.uid not in self.pending:
            raise ValueError("candidate is unknown or already scored")
        if not math.isfinite(score):
            raise ValueError("candidate reward must be finite")
        actual = self.pending.pop(candidate.uid)
        self.feedback.append(dict(uid=actual.uid, round=self.round, template_id=actual.template_id, embodiment=actual.value.tolist(), reward=float(score), rejected=bool(rejected)))

    def set_diagnostics(self, candidates, diagnostics):
        if any(len(values) != len(candidates) for values in diagnostics.values()):
            raise ValueError("diagnostic count must match population")
        self.diagnostics = {c.uid: {name: float(values[i]) for name, values in diagnostics.items()} for i, c in enumerate(candidates)}

    def end_round(self):
        if self.pending or not self.feedback:
            raise RuntimeError("a complete population must be scored before ending the round")
        with (self.output / "llm_feedback.jsonl").open("a") as stream:
            for row in self.feedback:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
        for template_id, template in enumerate(self.templates):
            rows = [row for row in self.feedback if row["template_id"] == template_id]
            rewards = np.asarray([row["reward"] for row in rows])
            self.history.append(dict(round=self.round, template_id=template_id, embodiment=template.tolist(), evaluations=len(rows), mean_reward=float(rewards.mean()), std_reward=float(rewards.std()), min_reward=float(rewards.min()), max_reward=float(rewards.max()), rejection_rate=float(np.mean([row["rejected"] for row in rows]))))
            if self.diagnostics:
                for name in next(iter(self.diagnostics.values())):
                    self.history[-1][name] = float(np.mean([self.diagnostics[row["uid"]][name] for row in rows]))
        self._write("llm_template_statistics.json", self.history)
        with (self.output / "llm_template_statistics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in self.history for key in row)))
            writer.writeheader()
            writer.writerows({**row, "embodiment": json.dumps(row["embodiment"])} for row in self.history)
        self.feedback = []
        self.diagnostics = {}
        self.round += 1

    def metrics(self):
        return {"llm/generated": self.generated, "llm/invalid_outputs": self.invalid, "llm/invalid_batches": self.invalid_batches, "llm/accepted": self.accepted, "llm/generation_seconds": self.generation_seconds, "llm/modeled_sampling_seconds": self.modeled_sampling_seconds, "llm/template_rows": len(self.history), "llm/prompt_tokens": self.prompt_tokens, **self.gateway.metrics()}

    def dump(self, path):
        self._write("llm_sampler.json", dict(model=self.args.llm_model, revision=self.resolved_revision, round=self.round, calls=self.calls, history=self.history, rng_state=self.rng.bit_generator.state, **self.metrics()))

    def close(self):
        self.gateway.close()
