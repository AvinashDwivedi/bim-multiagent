from __future__ import annotations

import json
from hashlib import sha256
from typing import Any
from uuid import uuid4

from .contracts import EvidenceArtifact


class EvidenceBudgetExhausted(RuntimeError):
    def __init__(self, phase: str, remaining: dict[str, int]) -> None:
        self.phase = phase
        self.remaining = remaining
        super().__init__(
            f"Evidence budget for phase '{phase}' is exhausted. Stop querying and submit the "
            f"best evidence-backed result. Remaining phase budgets: {remaining}."
        )


class EvidenceStore:
    def __init__(self, *, phase_limits: dict[str, int] | None = None, total_limit: int = 18) -> None:
        self._artifacts: list[EvidenceArtifact] = []
        self._phase = "investigation"
        self._phase_limits = phase_limits or {"investigation": total_limit}
        self._total_limit = total_limit
        self._cache: dict[str, EvidenceArtifact] = {}

    @property
    def artifacts(self) -> list[EvidenceArtifact]:
        return list(self._artifacts)

    @property
    def phase(self) -> str:
        return self._phase

    def set_phase(self, phase: str) -> None:
        if phase not in self._phase_limits:
            raise ValueError(f"Unknown evidence phase: {phase}")
        self._phase = phase

    def used(self, phase: str | None = None) -> int:
        if phase is None:
            return len(self._artifacts)
        return sum(item.phase == phase for item in self._artifacts)

    def remaining(self, phase: str | None = None) -> int:
        selected = phase or self._phase
        phase_remaining = self._phase_limits[selected] - self.used(selected)
        total_remaining = self._total_limit - len(self._artifacts)
        return max(0, min(phase_remaining, total_remaining))

    def remaining_by_phase(self) -> dict[str, int]:
        return {phase: self.remaining(phase) for phase in self._phase_limits}

    def require_capacity(self) -> None:
        if self.remaining() <= 0:
            raise EvidenceBudgetExhausted(self._phase, self.remaining_by_phase())

    @staticmethod
    def request_key(tool: str, data: dict[str, Any]) -> str:
        material = {
            key: value for key, value in data.items()
            if key not in {"purpose", "population", "entity_role", "spatial_scope",
                           "measurement_basis", "aggregation", "unit", "identity_key",
                           "inclusion_rules", "exclusion_rules"}
        }
        encoded = json.dumps(
            {"tool": tool, "request": material}, ensure_ascii=False, sort_keys=True, default=str
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def cached(self, key: str) -> EvidenceArtifact | None:
        return self._cache.get(key)

    def remember(self, key: str, artifact: EvidenceArtifact) -> None:
        self._cache[key] = artifact

    def add(self, *, tool: str, purpose: str, **data) -> EvidenceArtifact:
        self.require_capacity()
        artifact = EvidenceArtifact(
            artifact_id=f"{tool}-{uuid4().hex[:12]}",
            tool=tool,
            purpose=purpose,
            phase=self._phase,
            **data,
        )
        self._artifacts.append(artifact)
        return artifact
