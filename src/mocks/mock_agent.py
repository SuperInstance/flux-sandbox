"""
Mock agents for deterministic FLUX cooperative simulation.

Provides CooperativeMessage (unified message type), CooperativeResponse
(semantic alias for response messages), MockAgent (simulated fleet agent
with configurable behaviors and trust scoring), and MockTransport
(in-memory message router that bypasses git).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Unified message type
# ---------------------------------------------------------------------------

@dataclass
class CooperativeMessage:
    """Single message exchanged between cooperative agents.

    Attributes:
        msg_type:      ``"request"`` or ``"response"``.
        request_type:  Semantic operation name (e.g. ``"ping"``, ``"ask_bytecode"``).
        payload:       Arbitrary data carried by the message.
        sender:        Name of the originating agent.
        target:        Name of the intended recipient.
        message_id:    Globally unique identifier for this message.
        in_reply_to:   ``message_id`` of the message this responds to (empty for requests).
        timestamp:     Simulated wall-clock time (seconds).
        hop_count:     Number of network hops this message has traversed.
        metadata:      Extensible key/value bag for scenario-specific data.
    """

    msg_type: str = "request"
    request_type: str = ""
    payload: Any = None
    sender: str = ""
    target: str = ""
    message_id: str = ""
    in_reply_to: str = ""
    timestamp: float = 0.0
    hop_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Behavior type
# ---------------------------------------------------------------------------

AgentBehavior = Union[
    Callable[[CooperativeMessage], Optional[CooperativeMessage]],
    Dict[str, Callable[[CooperativeMessage], Optional[CooperativeMessage]]],
]


# ---------------------------------------------------------------------------
# Trust helpers
# ---------------------------------------------------------------------------

_DEFAULT_INITIAL_TRUST = 0.5
_TRUST_REWARD = 0.1
_TRUST_PENALTY = 0.2


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# MockAgent
# ---------------------------------------------------------------------------

class MockAgent:
    """Simulated agent that can send and receive cooperative tasks.

    The agent owns an inbox that the simulation harness delivers into.  When
    its :meth:`receive` method is called it consults *behavior* to produce a
    response (or ``None``).  Trust scores are updated automatically whenever a
    ``"response"``-type message is received.

    Parameters:
        name:         Unique agent identifier within the simulation.
        capabilities: List of capability tags this agent advertises.
        behavior:     Either a dict mapping *request_type* → handler callable,
                      or a single callable that handles every request.  Each
                      handler receives a :class:`CooperativeMessage` and may
                      return another :class:`CooperativeMessage` (the response)
                      or ``None`` (no response).
    """

    def __init__(
        self,
        name: str,
        capabilities: Optional[List[str]] = None,
        behavior: Optional[AgentBehavior] = None,
    ) -> None:
        self.name: str = name
        self.capabilities: List[str] = list(capabilities or [])
        self.inbox: List[CooperativeMessage] = []
        self.trust_scores: Dict[str, float] = {}
        self.task_handler: AgentBehavior = behavior or {}
        self._harness: Optional[Any] = None
        self._id_counter: int = 0

    # -- harness wiring -----------------------------------------------------

    def set_harness(self, harness: Any) -> None:
        """Bind this agent to a simulation harness (called by the harness)."""
        self._harness = harness

    # -- id generation ------------------------------------------------------

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"{self.name}_msg_{self._id_counter}"

    # -- receiving ----------------------------------------------------------

    def receive(self, message: CooperativeMessage) -> Optional[CooperativeMessage]:
        """Process an incoming message.

        * If *message* is a **response**, trust scores are updated and no
          further reply is generated.
        * If *message* is a **request**, the registered *behavior* handler is
          invoked and its return value (if any) is wrapped as a response.
        """
        if message.msg_type == "response":
            self._update_trust(message)
            return None

        # Resolve handler
        handler: Optional[Callable] = None
        if isinstance(self.task_handler, dict):
            handler = self.task_handler.get(message.request_type)
        elif callable(self.task_handler):
            handler = self.task_handler

        if handler is None:
            return None

        try:
            result = handler(message)
        except Exception as exc:
            result = CooperativeMessage(
                msg_type="response",
                request_type=message.request_type,
                payload={"status": "error", "error": str(exc)},
                sender=self.name,
                target=message.sender,
                message_id=self._next_id(),
                in_reply_to=message.message_id,
            )
            return result

        if result is not None and isinstance(result, CooperativeMessage):
            result.sender = self.name
            result.msg_type = "response"
            result.in_reply_to = message.message_id
            if not result.message_id:
                result.message_id = self._next_id()

        return result

    # -- sending ------------------------------------------------------------

    def send(self, target: str, task: Union[CooperativeMessage, Any]) -> bool:
        """Enqueue a message for delivery to *target* via the harness.

        If *task* is already a :class:`CooperativeMessage` its ``sender`` and
        ``target`` fields are overwritten.  Otherwise a request wrapper is
        created automatically.

        Returns ``True`` when the harness accepted the message.
        """
        if isinstance(task, CooperativeMessage):
            message = task
            message.sender = self.name
            message.target = target
            if not message.message_id:
                message.message_id = self._next_id()
        else:
            message = CooperativeMessage(
                msg_type="request",
                request_type="custom",
                payload=task,
                sender=self.name,
                target=target,
                message_id=self._next_id(),
            )

        if self._harness is not None:
            self._harness.enqueue_message(message)
            return True
        return False

    # -- inbox inspection ---------------------------------------------------

    def check_inbox(self) -> List[CooperativeMessage]:
        """Return a **copy** of the current inbox contents (non-destructive)."""
        return list(self.inbox)

    # -- trust management ---------------------------------------------------

    def _update_trust(self, response: CooperativeMessage) -> None:
        """Adjust trust for *response.sender* based on response status."""
        sender = response.sender
        if not sender:
            return

        current = self.trust_scores.get(sender, _DEFAULT_INITIAL_TRUST)

        if isinstance(response.payload, dict):
            status = response.payload.get("status", "unknown")
            if status == "success":
                current = _clamp(current + _TRUST_REWARD)
            else:
                current = _clamp(current - _TRUST_PENALTY)
        else:
            # Non-dict payload treated as opaque success
            current = _clamp(current + _TRUST_REWARD)

        self.trust_scores[sender] = current

    # -- repr ----------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        caps = ",".join(self.capabilities) or "none"
        return (
            f"MockAgent(name={self.name!r}, capabilities=[{caps}], "
            f"inbox_len={len(self.inbox)})"
        )


# ---------------------------------------------------------------------------
# CooperativeResponse — semantic alias for response messages
# ---------------------------------------------------------------------------

# ``receive()`` returns ``Optional[CooperativeMessage]`` where the response
# has ``msg_type == "response"``.  For clarity in type hints we expose a
# *CooperativeResponse* alias so callers can write:
#
#     response: Optional[CooperativeResponse] = agent.receive(msg)
#
CooperativeResponse = CooperativeMessage


# ---------------------------------------------------------------------------
# MockTransport — in-memory message router (no git required)
# ---------------------------------------------------------------------------

class MockTransport:
    """In-memory message transport that routes between registered agents.

    Acts as a simple publish/subscribe bus.  The simulation harness calls
    :meth:`deliver` and the transport places the message into the target
    agent's inbox.  No network, no git — pure in-memory routing.

    Usage::

        transport = MockTransport()
        transport.register(alice)
        transport.register(bob)
        transport.deliver(message)  # → lands in target agent's inbox
    """

    def __init__(self) -> None:
        self._agents: Dict[str, MockAgent] = {}

    # -- registration ---------------------------------------------------------

    def register(self, agent: MockAgent) -> None:
        """Register an agent so it can receive messages."""
        self._agents[agent.name] = agent

    def unregister(self, name: str) -> None:
        """Remove an agent from the routing table."""
        self._agents.pop(name, None)

    # -- delivery -------------------------------------------------------------

    def deliver(self, message: CooperativeMessage) -> bool:
        """Place *message* into the target agent's inbox.

        Returns ``True`` if the target agent is registered, ``False`` otherwise.
        """
        target = message.target
        agent = self._agents.get(target)
        if agent is None:
            return False
        agent.inbox.append(message)
        return True

    # -- query ----------------------------------------------------------------

    def is_registered(self, name: str) -> bool:
        return name in self._agents

    def registered_agents(self) -> List[str]:
        return list(self._agents.keys())

    # -- repr -----------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        names = ", ".join(self._agents.keys()) or "(none)"
        return f"MockTransport(agents=[{names}])"
