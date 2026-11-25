"""Serializable contracts for a future PPO/NGOpt co-design coordinator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class JsonRecord:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")


@dataclass(frozen=True)
class DesignCandidate(JsonRecord):
    candidate_id: str
    parameters: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyArtifact(JsonRecord):
    path: str
    environment_factory: str
    seed: int
    training_steps: int
    candidate_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult(JsonRecord):
    score: float
    candidate_id: Optional[str] = None
    policy_path: Optional[str] = None
    metrics: Mapping[str, float] = field(default_factory=dict)
