"""
Deterministic simulation harness for FLUX cooperative programs.

Provides SimulationHarness (the main execution engine), SimulationResult
(dataclass holding execution outcomes), and StepRecord (per-step trace
entry).  All non-determinism is channelled through a single PRNG so that
identical scenarios with the same seed produce identical results.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from src.mocks.mock_agent import CooperativeMessage, MockAgent, MockTransport
from src.failure.injector import FailureInjector, FailureType


# ---------------------------------------------------------------------------
# Trace / result types
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """Immutable record of a single simulation step.

    Attributes:
        step_num:          Zero-based step index.
        message_delivered: The message that was dequeued and delivered.
        response:          The agent's response (if any).
        agent_name:        Name of the agent that processed the message.
        failure:           :class:`FailureType` that was injected (or ``None``).
        wall_time_s:       Real wall-clock time taken for this step.
    """

    step_num: int
    message_delivered: Optional[CooperativeMessage]
    response: Optional[CooperativeMessage]
    agent_name: str
    failure: Optional[FailureType] = None
    wall_time_s: float = 0.0


@dataclass
class SimulationResult:
    """Summary of a completed simulation run.

    Attributes:
        scenario_name:       Name of the scenario that was executed.
        steps:               Ordered list of :class:`StepRecord` entries.
        messages:            All messages that were exchanged (flat list).
        final_trust_scores:  Mapping ``agent_name -> {peer: score}`` at end.
        timing:              Total wall-clock time in seconds.
        error:               Exception string if the run aborted, else ``None``.
    """

    scenario_name: str = ""
    steps: List[StepRecord] = field(default_factory=list)
    messages: List[CooperativeMessage] = field(default_factory=list)
    final_trust_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timing: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# SimulationHarness
# ---------------------------------------------------------------------------

class SimulationHarness:
    """Deterministic simulation environment for cooperative agent scenarios.

    The harness owns a set of agents, a PRNG for deterministic ordering,
    and a message queue.  Scenarios enqueue messages; the harness dequeues
    them one at a time, delivers to the target agent, and collects responses.

    Parameters:
        seed: Seed for the deterministic PRNG.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed: int = seed
        self._prng: random.Random = random.Random(seed)
        self.agents: Dict[str, MockAgent] = {}
        self.transport: MockTransport = MockTransport()
        self.message_queue: deque = deque()
        self.failure_injector: Optional[FailureInjector] = None
        self._step_counter: int = 0
        self._all_messages: List[CooperativeMessage] = []

    # -- agent management -----------------------------------------------------

    def add_agent(self, agent: MockAgent) -> None:
        """Register an agent with the harness and transport."""
        self.agents[agent.name] = agent
        self.transport.register(agent)
        agent.set_harness(self)

    def remove_agent(self, name: str) -> None:
        """Unregister an agent."""
        agent = self.agents.pop(name, None)
        if agent is not None:
            self.transport.unregister(name)
            agent.set_harness(None)

    # -- failure injection ----------------------------------------------------

    def inject_failure(self, step_num: int, failure_type: Union[FailureType, str]) -> None:
        """Schedule a failure at the given step number.

        *failure_type* may be a :class:`FailureType` enum member or its
        string value (e.g. ``"timeout"``).
        """
        if isinstance(failure_type, str):
            failure_type = FailureType(failure_type)
        if self.failure_injector is None:
            self.failure_injector = FailureInjector(seed=self.seed)
        self.failure_injector.inject(step_num, failure_type)

    # -- message queue --------------------------------------------------------

    def enqueue_message(self, message: CooperativeMessage) -> None:
        """Add a message to the queue for later delivery."""
        self.message_queue.append(message)

    def _enqueue_internal(self, message: CooperativeMessage) -> None:
        """Internal enqueue used by the harness for response messages."""
        self.message_queue.append(message)
        self._all_messages.append(message)

    # -- simulation execution -------------------------------------------------

    def run_scenario(
        self,
        scenario: Any,
        max_steps: int = 1000,
    ) -> SimulationResult:
        """Execute a scenario to completion.

        The scenario must expose an ``agents_config`` list of
        ``(name, capabilities, behavior)`` tuples and a ``steps`` list of
        :class:`CooperativeMessage` objects (or callables that return one)
        to seed the queue.

        Returns a :class:`SimulationResult` with the full execution trace.
        """
        t0 = time.monotonic()
        result = SimulationResult(scenario_name=getattr(scenario, "name", "<unnamed>"))

        # --- set up agents ---
        for agent_cfg in getattr(scenario, "agents_config", []):
            agent = MockAgent(
                name=agent_cfg[0],
                capabilities=agent_cfg[1],
                behavior=agent_cfg[2],
            )
            self.add_agent(agent)

        # --- seed initial messages ---
        for step in getattr(scenario, "steps", []):
            msg = step() if callable(step) else step
            self.enqueue_message(msg)
            self._all_messages.append(msg)

        # --- run steps ---
        step_count = 0
        while self.message_queue and step_count < max_steps:
            record = self.step()
            result.steps.append(record)
            step_count += 1

        # --- collect final trust scores ---
        for name, agent in self.agents.items():
            result.final_trust_scores[name] = dict(agent.trust_scores)

        result.messages = list(self._all_messages)
        result.timing = time.monotonic() - t0
        return result

    def step(self) -> StepRecord:
        """Dequeue the next message, deliver it, and return a :class:`StepRecord`.

        If a failure is scheduled for this step the message may be dropped,
        duplicated, delayed, or corrupted before delivery.
        """
        step_num = self._step_counter
        self._step_counter += 1
        t0 = time.monotonic()

        if not self.message_queue:
            return StepRecord(
                step_num=step_num,
                message_delivered=None,
                response=None,
                agent_name="",
                wall_time_s=0.0,
            )

        message = self.message_queue.popleft()
        agent_name = message.target

        # --- check for failures ---
        active_failure: Optional[FailureType] = None
        if self.failure_injector is not None:
            plans = self.failure_injector.get_failure_for_step(step_num, agent_name)
            if plans:
                active_failure = plans[0].failure_type

        if active_failure is not None:
            return self._apply_failure(step_num, message, agent_name, active_failure, t0)

        # --- normal delivery ---
        agent = self.agents.get(agent_name)
        if agent is None:
            return StepRecord(
                step_num=step_num,
                message_delivered=message,
                response=None,
                agent_name=agent_name,
                wall_time_s=time.monotonic() - t0,
            )

        response = agent.receive(message)
        if response is not None:
            self._enqueue_internal(response)

        return StepRecord(
            step_num=step_num,
            message_delivered=message,
            response=response,
            agent_name=agent_name,
            wall_time_s=time.monotonic() - t0,
        )

    # -- failure handling -----------------------------------------------------

    def _apply_failure(
        self,
        step_num: int,
        message: CooperativeMessage,
        agent_name: str,
        failure_type: FailureType,
        t0: float,
    ) -> StepRecord:
        """Apply a failure and return a record describing what happened."""

        if failure_type == FailureType.TIMEOUT:
            # Agent appears to not respond — no reply is generated.
            return StepRecord(
                step_num=step_num,
                message_delivered=message,
                response=None,
                agent_name=agent_name,
                failure=failure_type,
                wall_time_s=time.monotonic() - t0,
            )

        if failure_type == FailureType.MESSAGE_LOSS:
            # Message is silently dropped — never delivered.
            return StepRecord(
                step_num=step_num,
                message_delivered=None,
                response=None,
                agent_name=agent_name,
                failure=failure_type,
                wall_time_s=time.monotonic() - t0,
            )

        if failure_type == FailureType.DUPLICATE:
            # Message is delivered normally AND re-enqueued for later.
            agent = self.agents.get(agent_name)
            response = None
            if agent is not None:
                response = agent.receive(message)
                if response is not None:
                    self._enqueue_internal(response)
            # Re-enqueue a copy for a second delivery.
            dup = CooperativeMessage(
                msg_type=message.msg_type,
                request_type=message.request_type,
                payload=message.payload,
                sender=message.sender,
                target=message.target,
                message_id=f"{message.message_id}_dup",
                in_reply_to=message.in_reply_to,
                timestamp=message.timestamp,
                hop_count=message.hop_count + 1,
                metadata=dict(message.metadata),
            )
            self.message_queue.appendleft(dup)  # next in queue
            return StepRecord(
                step_num=step_num,
                message_delivered=message,
                response=response,
                agent_name=agent_name,
                failure=failure_type,
                wall_time_s=time.monotonic() - t0,
            )

        if failure_type == FailureType.DELAYED:
            # Message is moved to the back of the queue.
            self.message_queue.append(message)
            return StepRecord(
                step_num=step_num,
                message_delivered=None,
                response=None,
                agent_name=agent_name,
                failure=failure_type,
                wall_time_s=time.monotonic() - t0,
            )

        if failure_type == FailureType.CORRUPTED:
            # Payload is mangled but delivery proceeds.
            corrupted = CooperativeMessage(
                msg_type=message.msg_type,
                request_type=message.request_type,
                payload={"__corrupted__": True, "original": str(message.payload)},
                sender=message.sender,
                target=message.target,
                message_id=message.message_id,
                in_reply_to=message.in_reply_to,
                timestamp=message.timestamp,
                hop_count=message.hop_count,
                metadata=dict(message.metadata),
            )
            agent = self.agents.get(agent_name)
            response = None
            if agent is not None:
                response = agent.receive(corrupted)
                if response is not None:
                    self._enqueue_internal(response)
            return StepRecord(
                step_num=step_num,
                message_delivered=corrupted,
                response=response,
                agent_name=agent_name,
                failure=failure_type,
                wall_time_s=time.monotonic() - t0,
            )

        # Unknown failure — treat as no-op.
        return StepRecord(
            step_num=step_num,
            message_delivered=message,
            response=None,
            agent_name=agent_name,
            wall_time_s=time.monotonic() - t0,
        )

    # -- introspection --------------------------------------------------------

    def queue_size(self) -> int:
        return len(self.message_queue)

    def reset(self) -> None:
        """Clear all state for a fresh run (keeps seed)."""
        self._prng = random.Random(self.seed)
        self.agents.clear()
        self.transport = MockTransport()
        self.message_queue.clear()
        self._step_counter = 0
        self._all_messages.clear()
        self.failure_injector = None

    # -- repr -----------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        names = ", ".join(self.agents.keys()) or "(none)"
        return (
            f"SimulationHarness(seed={self.seed}, "
            f"agents=[{names}], queue_len={len(self.message_queue)})"
        )
