"""
Pre-built cooperation scenarios for deterministic FLUX sandbox testing.

Each scenario is a :class:`Scenario` dataclass that the SimulationHarness
can execute directly.  Factory functions return ready-to-run scenarios
for common cooperation patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.mocks.mock_agent import AgentBehavior, CooperativeMessage


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """A self-contained test scenario.

    Attributes:
        name:          Human-readable scenario name.
        description:   One-line description of what the scenario tests.
        agents_config: List of ``(name, capabilities, behavior)`` tuples.
                       Each tuple is unpacked into the ``MockAgent``
                       constructor by the harness.
        steps:         Ordered list of initial messages (or callables that
                       return messages) to seed the simulation queue.
    """

    name: str = ""
    description: str = ""
    agents_config: List[tuple] = field(default_factory=list)
    steps: List[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper — quick message factory
# ---------------------------------------------------------------------------

def _msg(
    sender: str,
    target: str,
    request_type: str,
    payload: Any = None,
    message_id: str = "",
) -> CooperativeMessage:
    """Create a CooperativeMessage with sensible defaults."""
    return CooperativeMessage(
        msg_type="request",
        request_type=request_type,
        payload=payload,
        sender=sender,
        target=target,
        message_id=message_id or f"{sender}_to_{target}_{request_type}",
    )


# ---------------------------------------------------------------------------
# Factory: ping scenario
# ---------------------------------------------------------------------------

def create_ping_scenario() -> Scenario:
    """Two agents — *alice* sends a ping to *bob*, bob replies pong.

    Agent *bob* has a single handler for ``"ping"`` that returns a pong
    response.
    """
    def bob_ping_handler(msg: CooperativeMessage) -> CooperativeMessage:
        return CooperativeMessage(
            msg_type="response",
            request_type="ping",
            payload={"status": "success", "data": "pong"},
            sender="bob",
            target=msg.sender,
            message_id="bob_pong",
            in_reply_to=msg.message_id,
        )

    return Scenario(
        name="ping",
        description="Alice sends ping to Bob, Bob replies pong",
        agents_config=[
            ("alice", ["sender"], {}),  # alice has no handlers — she initiates
            ("bob", ["responder"], {"ping": bob_ping_handler}),
        ],
        steps=[
            _msg("alice", "bob", "ping", {"echo": "hello"}),
        ],
    )


# ---------------------------------------------------------------------------
# Factory: ask-bytecode scenario
# ---------------------------------------------------------------------------

def create_ask_bytecode_scenario() -> Scenario:
    """Alice asks Bob to execute bytecode ``[MOVI R1,42, HALT]``.

    Bob's ``"ask_bytecode"`` handler simulates execution and returns the
    register state.
    """
    def bob_bytecode_handler(msg: CooperativeMessage) -> CooperativeMessage:
        instructions = msg.payload if isinstance(msg.payload, list) else []
        # Simulated execution: recognise MOVI and HALT
        registers: Dict[str, int] = {"R1": 0, "R2": 0, "R3": 0}
        for instr in instructions:
            parts = str(instr).upper().split()
            if parts and parts[0] == "MOVI" and len(parts) >= 2:
                # Handle both "R1 42" (space) and "R1,42" (comma) forms
                operand = parts[1]
                if "," in operand:
                    reg, val = operand.split(",", 1)
                else:
                    reg = operand
                    val = parts[2] if len(parts) >= 3 else "0"
                try:
                    registers[reg] = int(val)
                except (ValueError, KeyError):
                    pass
            elif parts and parts[0] == "HALT":
                break
        return CooperativeMessage(
            msg_type="response",
            request_type="ask_bytecode",
            payload={
                "status": "success",
                "registers": dict(registers),
                "instructions_executed": len(instructions),
            },
            sender="bob",
            target=msg.sender,
            message_id="bob_bytecode_result",
            in_reply_to=msg.message_id,
        )

    return Scenario(
        name="ask_bytecode",
        description="Alice asks Bob to execute [MOVI R1,42, HALT]",
        agents_config=[
            ("alice", ["asker"], {}),
            ("bob", ["executor"], {"ask_bytecode": bob_bytecode_handler}),
        ],
        steps=[
            _msg("alice", "bob", "ask_bytecode", ["MOVI R1,42", "HALT"]),
        ],
    )


# ---------------------------------------------------------------------------
# Factory: broadcast scenario
# ---------------------------------------------------------------------------

def create_broadcast_scenario() -> Scenario:
    """Three agents — *alice* broadcasts a task to *bob* and *carol*.

    Both bob and carol reply, and alice collects the results.
    """
    def _responder_reply(name: str) -> Callable[[CooperativeMessage], CooperativeMessage]:
        def handler(msg: CooperativeMessage) -> CooperativeMessage:
            return CooperativeMessage(
                msg_type="response",
                request_type="broadcast",
                payload={"status": "success", "from": name, "echo": msg.payload},
                sender=name,
                target=msg.sender,
                message_id=f"{name}_broadcast_ack",
                in_reply_to=msg.message_id,
            )
        return handler

    return Scenario(
        name="broadcast",
        description="Alice broadcasts to Bob and Carol, both respond",
        agents_config=[
            ("alice", ["coordinator"], {}),  # alice initiates broadcast
            ("bob", ["worker"], {"broadcast": _responder_reply("bob")}),
            ("carol", ["worker"], {"broadcast": _responder_reply("carol")}),
        ],
        steps=[
            _msg("alice", "bob", "broadcast", {"task": "report_status"}),
            _msg("alice", "carol", "broadcast", {"task": "report_status"}),
        ],
    )


# ---------------------------------------------------------------------------
# Factory: failure scenario (timeout)
# ---------------------------------------------------------------------------

def create_failure_scenario() -> Scenario:
    """Two agents — alice asks bob, but the harness injects a timeout at
    step 0 so bob never responds.

    The scenario's ``name`` carries a ``timeout_at_step`` metadata hint
    so the caller knows where to inject the failure.
    """
    def bob_slow_handler(msg: CooperativeMessage) -> CooperativeMessage:
        return CooperativeMessage(
            msg_type="response",
            request_type="ask",
            payload={"status": "success", "data": "would have replied"},
            sender="bob",
            target=msg.sender,
            message_id="bob_slow_reply",
            in_reply_to=msg.message_id,
        )

    return Scenario(
        name="failure_timeout",
        description="Alice asks Bob but a timeout is injected at step 0",
        agents_config=[
            ("alice", ["asker"], {}),
            ("bob", ["responder"], {"ask": bob_slow_handler}),
        ],
        steps=[
            _msg("alice", "bob", "ask", {"question": "what is 2+2?"}),
        ],
    )
