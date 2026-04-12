"""
Failure injection for deterministic FLUX cooperative simulation.

Provides FailureType enum and FailureInjector class that can selectively
inject network-level failures (timeouts, message loss, duplication, delays,
and corruption) at specific simulation steps or probabilistically.
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class FailureType(enum.Enum):
    """Types of failures that can be injected into a simulation."""

    TIMEOUT = "timeout"
    MESSAGE_LOSS = "message_loss"
    DUPLICATE = "duplicate"
    DELAYED = "delayed"
    CORRUPTED = "corrupted"


@dataclass
class FailurePlan:
    """A single planned failure injection.

    Attributes:
        step:         The simulation step number at which to inject.
        failure_type: What kind of failure to inject.
        target:       Optional agent name to target (None = any agent).
    """

    step: int
    failure_type: FailureType
    target: Optional[str] = None


class FailureInjector:
    """Deterministic failure injector for simulation harnesses.

    Failures can be scheduled explicitly by step number via :meth:`inject` or
    generated probabilistically via :meth:`maybe_inject`.  A bounded PRNG
    (seeded on construction) ensures reproducibility.

    Parameters:
        seed:         Seed for the internal PRNG.
        failure_rate: Probability [0.0, 1.0] of a random failure per step.
    """

    def __init__(self, seed: int = 42, failure_rate: float = 0.0) -> None:
        self._prng = random.Random(seed)
        self.failure_rate: float = max(0.0, min(1.0, failure_rate))
        self._planned: List[FailurePlan] = []
        self._applied_steps: set = set()

    # -- explicit scheduling -------------------------------------------------

    def inject(self, step: int, failure_type: FailureType, target: Optional[str] = None) -> None:
        """Schedule a failure at the given simulation *step*.

        If *target* is provided the failure will only affect messages destined
        for that agent.
        """
        self._planned.append(FailurePlan(step=step, failure_type=failure_type, target=target))

    # -- probabilistic injection ---------------------------------------------

    def maybe_inject(self, step: int, target: Optional[str] = None) -> Optional[FailureType]:
        """Roll the PRNG and possibly return a failure type for this step.

        Uses :attr:`failure_rate` as the per-step probability.  Returns the
        chosen :class:`FailureType` or ``None`` if no failure fires.
        """
        if self._prng.random() >= self.failure_rate:
            return None
        chosen = self._prng.choice(list(FailureType))
        self._planned.append(FailurePlan(step=step, failure_type=chosen, target=target))
        return chosen

    # -- query ----------------------------------------------------------------

    def get_failure_for_step(self, step: int, target: Optional[str] = None) -> List[FailurePlan]:
        """Return all planned failures matching *step* (and optionally *target*).

        Each plan is returned at most once; after retrieval it is moved to the
        ``_applied_steps`` set.
        """
        results: List[FailurePlan] = []
        remaining: List[FailurePlan] = []
        for plan in self._planned:
            if plan.step == step and (target is None or plan.target is None or plan.target == target):
                plan_key = (plan.step, plan.failure_type.value, plan.target)
                if plan_key not in self._applied_steps:
                    self._applied_steps.add(plan_key)
                    results.append(plan)
                    continue  # consumed — do not keep in remaining
            remaining.append(plan)
        self._planned = remaining
        return results

    def has_planned_failures(self) -> bool:
        """Return ``True`` if there are unplanned (future) failures."""
        return len(self._planned) > 0

    # -- repr -----------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FailureInjector(seed={self._prng.getstate()[0]}, "
            f"failure_rate={self.failure_rate}, "
            f"planned={len(self._planned)})"
        )
