"""
flux-sandbox / fleet_compat.py
Compatibility shim between sandbox simulation types and fleet-stdlib.

Maps the sandbox's internal FailureType / simulation outcome states to
fleet-standard :class:`Status` codes and :class:`FleetError` instances
so that simulation results can be consumed uniformly by fleet tooling.

Usage::

    from src.fleet_compat import failure_to_fleet_status, to_fleet_error

    # Map a FailureType injection to a fleet Status
    status = failure_to_fleet_status(FailureType.TIMEOUT)
    # -> Status.TIMEOUT

    # Wrap a simulation error
    err = to_fleet_error(sim_exception, step_num=42)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Import from fleet-stdlib (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from flux_fleet_stdlib.errors import ErrorCode, FleetError, Severity, fleet_error  # type: ignore[import-untyped]
    from flux_fleet_stdlib.status import Status, status_for_error_code  # type: ignore[import-untyped]
    _STDLIB_AVAILABLE = True
except ImportError:
    _STDLIB_AVAILABLE = False

    # Minimal stubs so the module loads even without fleet-stdlib installed.
    class ErrorCode:  # type: ignore[no-redef]
        COOP_TIMEOUT = "COOP_TIMEOUT"
        TRANSPORT_NETWORK_ERROR = "TRANSPORT_NETWORK_ERROR"
        COOP_TRANSPORT_FAILURE = "COOP_TRANSPORT_FAILURE"
        COOP_UNKNOWN_REQUEST = "COOP_UNKNOWN_REQUEST"
        SECURITY_SANDBOX_VIOLATION = "SECURITY_SANDBOX_VIOLATION"

    class FleetError(Exception):  # type: ignore[no-redef]
        def __init__(self, code: str, message: str, **kw: Any) -> None:
            self.code = code
            self.message = message
            super().__init__(f"[{code}] {message}")

    class Severity:  # type: ignore[no-redef]
        ERROR = "ERROR"
        WARNING = "WARNING"

    def fleet_error(code: str, message: str, **kw: Any) -> FleetError:  # type: ignore[misc]
        return FleetError(code, message)

    class Status:  # type: ignore[no-redef]
        SUCCESS = "SUCCESS"
        PENDING = "PENDING"
        TIMEOUT = "TIMEOUT"
        ERROR = "ERROR"
        CANCELLED = "CANCELLED"
        PARTIAL = "PARTIAL"
        REFUSED = "REFUSED"

    def status_for_error_code(code: str) -> "Status":  # type: ignore[misc]
        return Status.ERROR


# ---------------------------------------------------------------------------
# Import sandbox-internal types (used for type annotations & mapping)
# ---------------------------------------------------------------------------

from src.failure.injector import FailureType  # noqa: E402


# ---------------------------------------------------------------------------
# FailureType -> fleet Status mapping
# ---------------------------------------------------------------------------

FAILURE_TO_STATUS: Dict[FailureType, Status] = {
    FailureType.TIMEOUT: Status.TIMEOUT,
    FailureType.MESSAGE_LOSS: Status.ERROR,
    FailureType.DUPLICATE: Status.PARTIAL,
    FailureType.DELAYED: Status.PENDING,
    FailureType.CORRUPTED: Status.ERROR,
}

FAILURE_TO_ERROR_CODE: Dict[FailureType, str] = {
    FailureType.TIMEOUT: ErrorCode.COOP_TIMEOUT,
    FailureType.MESSAGE_LOSS: ErrorCode.TRANSPORT_NETWORK_ERROR,
    FailureType.DUPLICATE: ErrorCode.COOP_TRANSPORT_FAILURE,
    FailureType.DELAYED: ErrorCode.COOP_TIMEOUT,
    FailureType.CORRUPTED: ErrorCode.COOP_DESERIALIZATION_ERROR
    if hasattr(ErrorCode, "COOP_DESERIALIZATION_ERROR")
    else ErrorCode.COOP_TRANSPORT_FAILURE,
}


def failure_to_fleet_status(ft: FailureType) -> Status:
    """Map a :class:`FailureType` to a fleet :class:`Status`.

    This is the primary bridge between the sandbox's failure-injection
    taxonomy and the fleet's unified status codes.
    """
    return FAILURE_TO_STATUS.get(ft, Status.ERROR)


def failure_to_error_code(ft: FailureType) -> str:
    """Map a :class:`FailureType` to a fleet :class:`ErrorCode` value.

    Used when a failure needs to be wrapped in a :class:`FleetError`.
    """
    code = FAILURE_TO_ERROR_CODE.get(ft)
    return code if code else ErrorCode.COOP_UNKNOWN_REQUEST


# ---------------------------------------------------------------------------
# SimulationResult -> fleet Status
# ---------------------------------------------------------------------------

def simulation_result_to_status(
    has_error: bool,
    completed_all_steps: bool,
    max_steps_reached: bool,
) -> Status:
    """Derive a fleet :class:`Status` from a simulation outcome summary.

    Parameters:
        has_error:         True if the simulation raised an exception.
        completed_all_steps: True if the message queue drained naturally.
        max_steps_reached: True if the harness stopped due to the step cap.

    Returns:
        A :class:`Status` value.
    """
    if has_error:
        return Status.ERROR
    if max_steps_reached and not completed_all_steps:
        return Status.PARTIAL
    if completed_all_steps:
        return Status.SUCCESS
    return Status.CANCELLED


# ---------------------------------------------------------------------------
# FleetError helpers
# ---------------------------------------------------------------------------

def to_fleet_error(
    exc: Exception,
    *,
    step_num: Optional[int] = None,
    agent_name: str = "",
    failure_type: Optional[FailureType] = None,
    default_code: str = "COOP_UNKNOWN_REQUEST",
    source_repo: str = "flux-sandbox",
    **extra_context: Any,
) -> FleetError:
    """Wrap any exception in a :class:`FleetError`.

    If *exc* is already a :class:`FleetError`, it is returned unchanged.

    When a *failure_type* is provided, the error code is derived from the
    standard failure-to-code mapping table.

    Parameters:
        exc:            The exception to wrap.
        step_num:       Simulation step at which the error occurred.
        agent_name:     Agent that caused / observed the error.
        failure_type:   Optional :class:`FailureType` for code lookup.
        default_code:   Fleet error code if no mapping applies.
        source_repo:    Repo identifier stamped on the error.
        **extra_context: Additional key-value pairs for ``context``.

    Returns:
        A :class:`FleetError` instance.
    """
    if isinstance(exc, FleetError):
        return exc

    # Determine the code from failure_type if available.
    if failure_type is not None:
        code = failure_to_error_code(failure_type)
    else:
        code = default_code

    context: Dict[str, Any] = {"original_type": type(exc).__name__}
    if step_num is not None:
        context["step_num"] = step_num
    if agent_name:
        context["agent_name"] = agent_name
    if failure_type is not None:
        context["failure_type"] = failure_type.value
    context.update(extra_context)

    return FleetError(
        code=code,
        message=str(exc),
        severity=Severity.ERROR,
        source_repo=source_repo,
        source_agent=agent_name,
        context=context,
    )


def simulation_error(
    code: str,
    message: str,
    *,
    scenario_name: str = "",
    step_num: Optional[int] = None,
    agent_name: str = "",
    severity: str = Severity.ERROR,
    **extra_context: Any,
) -> FleetError:
    """Create a :class:`FleetError` specifically tagged for simulation contexts.

    Convenience wrapper that always sets ``source_repo`` to
    ``"flux-sandbox"`` and injects ``scenario_name`` into context.
    """
    context: Dict[str, Any] = {}
    if scenario_name:
        context["scenario_name"] = scenario_name
    if step_num is not None:
        context["step_num"] = step_num
    if agent_name:
        context["agent_name"] = agent_name
    context.update(extra_context)

    return fleet_error(
        code=code,
        message=message,
        severity=severity,
        source_repo="flux-sandbox",
        source_agent=agent_name,
        **context,
    )


# ---------------------------------------------------------------------------
# StepRecord enrichment
# ---------------------------------------------------------------------------

def enrich_step_record_fleet(
    step_dict: Dict[str, Any],
    failure: Optional[FailureType] = None,
) -> Dict[str, Any]:
    """Add fleet-standard ``fleet_status`` and ``fleet_error_code`` fields
    to a step record dict (e.g. for JSON export).

    This is a *non-mutating* helper — it returns a new dict.
    """
    result = dict(step_dict)
    if failure is not None:
        result["fleet_status"] = failure_to_fleet_status(failure).value
        result["fleet_error_code"] = failure_to_error_code(failure)
    else:
        result["fleet_status"] = Status.SUCCESS.value
        result["fleet_error_code"] = ""
    return result
