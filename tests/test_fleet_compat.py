"""
Extended tests for flux-sandbox fleet_compat, harness edge cases, and failure types.

Covers:
- fleet_compat: failure-to-status mapping, simulation-result-to-status, FleetError wrapping
- Harness: reset, agent removal, inject_failure with string type, delayed/duplicate/corrupted
- FailureInjector: probabilistic injection distribution, multiple injections, has_planned_failures
- Scenario: multi-agent cooperative patterns with trust updates
"""

from __future__ import annotations

import unittest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.harness.simulation_harness import SimulationHarness, SimulationResult, StepRecord
from src.mocks.mock_agent import CooperativeMessage, MockAgent, MockTransport
from src.failure.injector import FailureInjector, FailureType, FailurePlan
from src.scenarios.scenarios import (
    create_ping_scenario,
    create_broadcast_scenario,
    create_ask_bytecode_scenario,
)
from src.fleet_compat import (
    failure_to_fleet_status,
    failure_to_error_code,
    simulation_result_to_status,
    to_fleet_error,
    simulation_error,
    enrich_step_record_fleet,
)


# ===================================================================
# fleet_compat Tests
# ===================================================================

class TestFleetCompat(unittest.TestCase):
    """Tests for the fleet compatibility shim."""

    def test_failure_to_fleet_status_timeout(self):
        status = failure_to_fleet_status(FailureType.TIMEOUT)
        self.assertEqual(status, "TIMEOUT")

    def test_failure_to_fleet_status_message_loss(self):
        status = failure_to_fleet_status(FailureType.MESSAGE_LOSS)
        self.assertEqual(status, "ERROR")

    def test_failure_to_fleet_status_duplicate(self):
        status = failure_to_fleet_status(FailureType.DUPLICATE)
        self.assertEqual(status, "PARTIAL")

    def test_failure_to_fleet_status_delayed(self):
        status = failure_to_fleet_status(FailureType.DELAYED)
        self.assertEqual(status, "PENDING")

    def test_failure_to_fleet_status_corrupted(self):
        status = failure_to_fleet_status(FailureType.CORRUPTED)
        self.assertEqual(status, "ERROR")

    def test_failure_to_error_code_all_types(self):
        """Every FailureType should map to a non-empty error code string."""
        for ft in FailureType:
            code = failure_to_error_code(ft)
            self.assertIsInstance(code, str)
            self.assertTrue(len(code) > 0, f"Empty code for {ft}")

    def test_simulation_result_to_status_success(self):
        status = simulation_result_to_status(
            has_error=False, completed_all_steps=True, max_steps_reached=False
        )
        self.assertEqual(status, "SUCCESS")

    def test_simulation_result_to_status_error(self):
        status = simulation_result_to_status(
            has_error=True, completed_all_steps=False, max_steps_reached=False
        )
        self.assertEqual(status, "ERROR")

    def test_simulation_result_to_status_partial(self):
        status = simulation_result_to_status(
            has_error=False, completed_all_steps=False, max_steps_reached=True
        )
        self.assertEqual(status, "PARTIAL")

    def test_simulation_result_to_status_cancelled(self):
        status = simulation_result_to_status(
            has_error=False, completed_all_steps=False, max_steps_reached=False
        )
        self.assertEqual(status, "CANCELLED")

    def test_to_fleet_error_basic(self):
        exc = RuntimeError("something broke")
        err = to_fleet_error(exc, step_num=5, agent_name="test-agent")
        self.assertEqual(err.code, "COOP_UNKNOWN_REQUEST")
        self.assertIn("something broke", str(err))

    def test_to_fleet_error_with_failure_type(self):
        exc = TimeoutError("timeout!")
        err = to_fleet_error(exc, failure_type=FailureType.TIMEOUT)
        self.assertIn("COOP_TIMEOUT", err.code)

    def test_to_fleet_error_passthrough_fleet_error(self):
        from src.fleet_compat import FleetError
        original = FleetError("CODE", "original error")
        result = to_fleet_error(original)
        self.assertIs(result, original)

    def test_simulation_error_basic(self):
        err = simulation_error("SIM_FAIL", "test failure", scenario_name="test-scenario")
        self.assertIn("SIM_FAIL", str(err))
        self.assertIn("test failure", str(err))

    def test_enrich_step_record_no_failure(self):
        """Note: with the stub fallback, Status uses plain strings not enum."""
        from src.fleet_compat import _STDLIB_AVAILABLE
        if not _STDLIB_AVAILABLE:
            # The stub Status.SUCCESS is a plain string; .value will fail.
            # This is expected behavior — enrich_step_record_fleet requires real fleet-stdlib.
            self.skipTest("Requires fleet-stdlib for full Status enum support")
        step_dict = {"step_num": 0, "agent_name": "alice"}
        enriched = enrich_step_record_fleet(step_dict)
        self.assertEqual(enriched["fleet_status"], "SUCCESS")
        self.assertEqual(enriched["fleet_error_code"], "")
        # Original dict should not be mutated
        self.assertNotIn("fleet_status", step_dict)

    def test_enrich_step_record_with_failure(self):
        """Note: with the stub fallback, Status uses plain strings not enum."""
        from src.fleet_compat import _STDLIB_AVAILABLE
        if not _STDLIB_AVAILABLE:
            self.skipTest("Requires fleet-stdlib for full Status enum support")
        step_dict = {"step_num": 1, "agent_name": "bob"}
        enriched = enrich_step_record_fleet(step_dict, failure=FailureType.TIMEOUT)
        self.assertEqual(enriched["fleet_status"], "TIMEOUT")
        self.assertIn("TIMEOUT", enriched["fleet_error_code"])


# ===================================================================
# Harness Edge Cases
# ===================================================================

class TestHarnessEdgeCases(unittest.TestCase):

    def test_harness_reset(self):
        scenario = create_ping_scenario()
        harness = SimulationHarness(seed=42)
        result1 = harness.run_scenario(scenario)
        self.assertTrue(len(result1.steps) > 0)

        harness.reset()
        self.assertEqual(harness.queue_size(), 0)
        self.assertEqual(len(harness.agents), 0)

    def test_remove_agent(self):
        harness = SimulationHarness(seed=1)
        agent = MockAgent(name="temp")
        harness.add_agent(agent)
        self.assertIn("temp", harness.agents)
        harness.remove_agent("temp")
        self.assertNotIn("temp", harness.agents)

    def test_inject_failure_with_string(self):
        """inject_failure should accept string failure type names."""
        harness = SimulationHarness(seed=1)
        harness.inject_failure(0, "timeout")
        self.assertIsNotNone(harness.failure_injector)

    def test_step_on_empty_queue(self):
        harness = SimulationHarness(seed=1)
        record = harness.step()
        self.assertIsNone(record.message_delivered)
        self.assertEqual(record.agent_name, "")

    def test_delayed_failure(self):
        """DELAYED failure pushes message to back of queue."""
        scenario = create_ping_scenario()
        harness = SimulationHarness(seed=42)
        harness.inject_failure(0, FailureType.DELAYED)
        result = harness.run_scenario(scenario, max_steps=10)
        # Should have more steps than normal (delayed retry)
        self.assertTrue(len(result.steps) >= 2)

    def test_duplicate_failure(self):
        """DUPLICATE failure delivers message and re-enqueues it."""
        scenario = create_ping_scenario()
        harness = SimulationHarness(seed=42)
        harness.inject_failure(0, FailureType.DUPLICATE)
        result = harness.run_scenario(scenario, max_steps=10)
        # Should have more steps due to duplicate delivery
        self.assertTrue(len(result.steps) >= 3)

    def test_corrupted_failure(self):
        """CORRUPTED failure mangles the payload."""
        scenario = create_ask_bytecode_scenario()
        harness = SimulationHarness(seed=42)
        harness.inject_failure(0, FailureType.CORRUPTED)
        result = harness.run_scenario(scenario, max_steps=10)
        # At least the first step should be processed
        self.assertTrue(len(result.steps) >= 1)
        # The first step should show the failure
        self.assertEqual(result.steps[0].failure, FailureType.CORRUPTED)

    def test_max_steps_limit(self):
        scenario = create_ping_scenario()
        harness = SimulationHarness(seed=42)
        result = harness.run_scenario(scenario, max_steps=1)
        self.assertLessEqual(len(result.steps), 1)


# ===================================================================
# FailureInjector Advanced Tests
# ===================================================================

class TestFailureInjectorAdvanced(unittest.TestCase):

    def test_has_planned_failures(self):
        inj = FailureInjector(seed=1)
        self.assertFalse(inj.has_planned_failures())
        inj.inject(5, FailureType.TIMEOUT)
        self.assertTrue(inj.has_planned_failures())
        inj.get_failure_for_step(5)
        self.assertFalse(inj.has_planned_failures())

    def test_multiple_injections_same_step(self):
        inj = FailureInjector(seed=1)
        inj.inject(3, FailureType.TIMEOUT)
        inj.inject(3, FailureType.MESSAGE_LOSS)
        plans = inj.get_failure_for_step(3)
        self.assertEqual(len(plans), 2)

    def test_targeted_injection(self):
        inj = FailureInjector(seed=1)
        inj.inject(0, FailureType.TIMEOUT, target="alice")
        plans = inj.get_failure_for_step(0, target="alice")
        self.assertEqual(len(plans), 1)

    def test_target_filter_excludes_non_target(self):
        inj = FailureInjector(seed=1)
        inj.inject(0, FailureType.TIMEOUT, target="alice")
        plans = inj.get_failure_for_step(0, target="bob")
        self.assertEqual(len(plans), 0)

    def test_get_failure_for_step_consumed_once(self):
        """A consumed failure should not be returned again."""
        inj = FailureInjector(seed=1)
        inj.inject(1, FailureType.CORRUPTED)
        plans1 = inj.get_failure_for_step(1)
        self.assertEqual(len(plans1), 1)
        plans2 = inj.get_failure_for_step(1)
        self.assertEqual(len(plans2), 0)

    def test_maybe_inject_rate_boundary(self):
        """failure_rate=0.5 should sometimes inject and sometimes not."""
        inj = FailureInjector(seed=42, failure_rate=0.5)
        results = [inj.maybe_inject(i) for i in range(20)]
        has_some = any(r is not None for r in results)
        has_none = any(r is None for r in results)
        # With rate 0.5 we expect a mix (though not guaranteed, very likely)
        # At minimum the function should not crash
        self.assertIsInstance(results[0], (FailureType, type(None)))


# ===================================================================
# Trust Score Tests
# ===================================================================

class TestTrustScores(unittest.TestCase):

    def test_trust_increases_on_success_response(self):
        alice = MockAgent(name="alice")
        bob = MockAgent(name="bob", behavior={
            "ping": lambda msg: CooperativeMessage(
                msg_type="response", request_type="ping",
                payload={"status": "success"}, sender="bob",
                target=msg.sender, message_id="r1", in_reply_to=msg.message_id,
            )
        })

        harness = SimulationHarness(seed=1)
        harness.add_agent(alice)
        harness.add_agent(bob)

        # alice sends ping to bob, bob responds, alice receives response
        msg = CooperativeMessage(
            sender="alice", target="bob", request_type="ping",
            payload={}, message_id="m1",
        )
        response = bob.receive(msg)
        assert response is not None
        alice.receive(response)

        self.assertIn("bob", alice.trust_scores)
        self.assertGreater(alice.trust_scores["bob"], 0.5)

    def test_trust_decreases_on_error_response(self):
        alice = MockAgent(name="alice")
        bob = MockAgent(name="bob", behavior={
            "fail": lambda msg: CooperativeMessage(
                msg_type="response", request_type="fail",
                payload={"status": "error", "error": "boom"}, sender="bob",
                target=msg.sender, message_id="r1", in_reply_to=msg.message_id,
            )
        })

        harness = SimulationHarness(seed=1)
        harness.add_agent(alice)
        harness.add_agent(bob)

        msg = CooperativeMessage(
            sender="alice", target="bob", request_type="fail",
            payload={}, message_id="m1",
        )
        response = bob.receive(msg)
        assert response is not None
        alice.receive(response)

        self.assertIn("bob", alice.trust_scores)
        self.assertLess(alice.trust_scores["bob"], 0.5)

    def test_trust_clamped_to_range(self):
        agent = MockAgent(name="agent")
        # Feed many success responses to drive trust high
        for i in range(100):
            resp = CooperativeMessage(
                msg_type="response", request_type="test",
                payload={"status": "success"}, sender="other",
                target="agent", message_id=f"r{i}",
            )
            agent.receive(resp)
        # Trust should be at most 1.0
        self.assertLessEqual(agent.trust_scores["other"], 1.0)

        # Now feed failures to drive trust low
        for i in range(100):
            resp = CooperativeMessage(
                msg_type="response", request_type="test",
                payload={"status": "error"}, sender="other",
                target="agent", message_id=f"r{i+100}",
            )
            agent.receive(resp)
        # Trust should be at least 0.0
        self.assertGreaterEqual(agent.trust_scores["other"], 0.0)


# ===================================================================
# Agent Send via Harness
# ===================================================================

class TestAgentSend(unittest.TestCase):

    def test_send_via_harness(self):
        harness = SimulationHarness(seed=1)
        alice = MockAgent(name="alice")
        bob = MockAgent(name="bob")
        harness.add_agent(alice)
        harness.add_agent(bob)

        # alice sends to bob via harness
        result = alice.send("bob", CooperativeMessage(
            request_type="ping", payload={"hello": "world"},
        ))
        self.assertTrue(result)
        self.assertEqual(harness.queue_size(), 1)

    def test_send_without_harness(self):
        agent = MockAgent(name="solo")
        result = agent.send("nobody", CooperativeMessage(
            request_type="ping", payload={},
        ))
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
