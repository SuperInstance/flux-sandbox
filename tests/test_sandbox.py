"""
Unit tests for the flux-sandbox simulation harness.

Covers: ping, ask/respond, broadcast, deterministic reproduction, and
failure injection (timeout).  All tests use only stdlib ``unittest``.
"""

from __future__ import annotations

import sys
import os
import unittest

# Ensure project root is on sys.path so ``src.xxx`` relative imports resolve.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.harness.simulation_harness import SimulationHarness, SimulationResult
from src.mocks.mock_agent import CooperativeMessage, MockAgent, MockTransport
from src.scenarios.scenarios import (
    Scenario,
    create_ping_scenario,
    create_ask_bytecode_scenario,
    create_broadcast_scenario,
    create_failure_scenario,
)
from src.failure.injector import FailureInjector, FailureType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_message_ids(result: SimulationResult) -> list:
    """Return a list of (sender, target, request_type, msg_type) tuples."""
    out = []
    for msg in result.messages:
        out.append((msg.sender, msg.target, msg.request_type, msg.msg_type))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPingScenario(unittest.TestCase):
    """Alice sends ping → Bob replies pong."""

    def test_ping_scenario(self):
        scenario = create_ping_scenario()
        harness = SimulationHarness(seed=123)
        result = harness.run_scenario(scenario)

        # The scenario should complete in 2 steps:
        #   step 0 — alice→bob (ping request)
        #   step 1 — bob→alice (pong response)
        self.assertEqual(result.scenario_name, "ping")
        self.assertEqual(len(result.steps), 2)

        # Step 0: ping delivered to bob
        s0 = result.steps[0]
        self.assertEqual(s0.agent_name, "bob")
        self.assertEqual(s0.message_delivered.request_type, "ping")

        # Step 1: response delivered to alice (no reply — alice just updates trust)
        s1 = result.steps[1]
        self.assertEqual(s1.agent_name, "alice")
        self.assertIsNotNone(s1.message_delivered)
        self.assertEqual(s1.message_delivered.msg_type, "response")
        self.assertEqual(s1.message_delivered.payload["data"], "pong")

    def test_ping_trust_update(self):
        """Alice's trust in Bob should increase after receiving pong."""
        scenario = create_ping_scenario()
        harness = SimulationHarness(seed=99)
        result = harness.run_scenario(scenario)

        # After the response, alice should have a trust score for bob > 0.5
        alice_trust = result.final_trust_scores.get("alice", {})
        self.assertIn("bob", alice_trust)
        self.assertGreater(alice_trust["bob"], 0.5)


class TestAskRespondScenario(unittest.TestCase):
    """Alice asks Bob to execute bytecode [MOVI R1,42, HALT]."""

    def test_ask_respond_scenario(self):
        scenario = create_ask_bytecode_scenario()
        harness = SimulationHarness(seed=456)
        result = harness.run_scenario(scenario)

        # 2 steps: request then response
        self.assertEqual(len(result.steps), 2)

        # The response message should contain register state with R1=42
        response_step = result.steps[1]
        self.assertIsNotNone(response_step.message_delivered)
        self.assertEqual(response_step.message_delivered.msg_type, "response")
        payload = response_step.message_delivered.payload
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["registers"]["R1"], 42)
        self.assertEqual(payload["instructions_executed"], 2)


class TestBroadcastScenario(unittest.TestCase):
    """Alice broadcasts to Bob and Carol; both respond."""

    def test_broadcast_scenario(self):
        scenario = create_broadcast_scenario()
        harness = SimulationHarness(seed=789)
        result = harness.run_scenario(scenario)

        # 4 steps: 2 requests (bob, carol) + 2 responses (alice, alice)
        self.assertEqual(len(result.steps), 4)

        # Check that alice receives responses from both bob and carol
        # (responses are delivered as messages; alice doesn't reply to them)
        alice_responses = [
            s.message_delivered
            for s in result.steps
            if s.agent_name == "alice" and s.message_delivered is not None
        ]
        self.assertEqual(len(alice_responses), 2)

        responders = {r.payload["from"] for r in alice_responses if r.msg_type == "response"}
        self.assertEqual(responders, {"bob", "carol"})

    def test_broadcast_trust_scores(self):
        """Alice should have trust entries for both bob and carol."""
        scenario = create_broadcast_scenario()
        harness = SimulationHarness(seed=101)
        result = harness.run_scenario(scenario)

        alice_trust = result.final_trust_scores.get("alice", {})
        self.assertIn("bob", alice_trust)
        self.assertIn("carol", alice_trust)


class TestDeterministicReproduction(unittest.TestCase):
    """Same seed → identical result across two runs."""

    def test_deterministic_reproduction(self):
        scenario = create_broadcast_scenario()
        seed = 314159

        harness_a = SimulationHarness(seed=seed)
        result_a = harness_a.run_scenario(scenario)

        harness_b = SimulationHarness(seed=seed)
        result_b = harness_b.run_scenario(scenario)

        # Same number of steps
        self.assertEqual(len(result_a.steps), len(result_b.steps))

        # Every step should match: same agent, same message, same response
        for sa, sb in zip(result_a.steps, result_b.steps):
            self.assertEqual(sa.step_num, sb.step_num)
            self.assertEqual(sa.agent_name, sb.agent_name)
            self.assertEqual(sa.failure, sb.failure)

            if sa.message_delivered is not None and sb.message_delivered is not None:
                self.assertEqual(
                    sa.message_delivered.request_type,
                    sb.message_delivered.request_type,
                )
            else:
                self.assertIsNone(sa.message_delivered)
                self.assertIsNone(sb.message_delivered)

            if sa.response is not None and sb.response is not None:
                self.assertEqual(sa.response.payload, sb.response.payload)
            else:
                self.assertIsNone(sa.response)
                self.assertIsNone(sb.response)

        # Trust scores must match
        self.assertEqual(result_a.final_trust_scores, result_b.final_trust_scores)

    def test_different_seeds_differ(self):
        """Different seeds should still produce the same result for a
        fully deterministic scenario (no randomness in handlers)."""
        scenario = create_ping_scenario()

        harness_a = SimulationHarness(seed=1)
        result_a = harness_a.run_scenario(scenario)

        harness_b = SimulationHarness(seed=2)
        result_b = harness_b.run_scenario(scenario)

        # Same deterministic outcome regardless of seed
        self.assertEqual(len(result_a.steps), len(result_b.steps))
        self.assertEqual(
            result_a.steps[0].message_delivered.request_type,
            result_b.steps[0].message_delivered.request_type,
        )


class TestFailureInjectionTimeout(unittest.TestCase):
    """Inject a TIMEOUT failure so Bob never responds."""

    def test_failure_injection_timeout(self):
        scenario = create_failure_scenario()
        harness = SimulationHarness(seed=555)

        # Inject timeout at step 0 (the request from alice→bob)
        harness.inject_failure(0, FailureType.TIMEOUT)

        result = harness.run_scenario(scenario)

        # Only 1 step — the request is delivered but bob times out
        self.assertEqual(len(result.steps), 1)

        s0 = result.steps[0]
        self.assertEqual(s0.agent_name, "bob")
        self.assertEqual(s0.failure, FailureType.TIMEOUT)
        self.assertIsNone(s0.response)

        # Alice should NOT have a trust score for bob (no response received)
        alice_trust = result.final_trust_scores.get("alice", {})
        self.assertNotIn("bob", alice_trust)

    def test_failure_injection_message_loss(self):
        scenario = create_ping_scenario()
        harness = SimulationHarness(seed=777)

        # Inject message loss at step 0 — message never delivered
        harness.inject_failure(0, FailureType.MESSAGE_LOSS)

        result = harness.run_scenario(scenario)

        # Only 1 step recorded, message_delivered is None
        self.assertEqual(len(result.steps), 1)
        self.assertIsNone(result.steps[0].message_delivered)
        self.assertEqual(result.steps[0].failure, FailureType.MESSAGE_LOSS)


class TestMockTransport(unittest.TestCase):
    """Lightweight sanity checks for MockTransport."""

    def test_register_and_deliver(self):
        transport = MockTransport()
        alice = MockAgent(name="alice")
        bob = MockAgent(name="bob")

        transport.register(alice)
        transport.register(bob)

        msg = CooperativeMessage(
            sender="alice",
            target="bob",
            request_type="ping",
            payload="hi",
        )
        delivered = transport.deliver(msg)

        self.assertTrue(delivered)
        self.assertEqual(len(bob.inbox), 1)
        self.assertEqual(bob.inbox[0].payload, "hi")

    def test_deliver_to_unknown(self):
        transport = MockTransport()
        msg = CooperativeMessage(
            sender="alice",
            target="ghost",
            request_type="ping",
        )
        self.assertFalse(transport.deliver(msg))


class TestFailureInjector(unittest.TestCase):
    """Direct unit tests for FailureInjector."""

    def test_explicit_injection(self):
        inj = FailureInjector(seed=1)
        inj.inject(0, FailureType.TIMEOUT)
        inj.inject(3, FailureType.MESSAGE_LOSS)

        plans = inj.get_failure_for_step(0)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].failure_type, FailureType.TIMEOUT)

        plans = inj.get_failure_for_step(3)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].failure_type, FailureType.MESSAGE_LOSS)

        # Step 5 has nothing
        plans = inj.get_failure_for_step(5)
        self.assertEqual(len(plans), 0)

    def test_maybe_inject_with_high_rate(self):
        inj = FailureInjector(seed=42, failure_rate=1.0)
        result = inj.maybe_inject(0)
        # With 100% rate, should always return a failure type
        self.assertIsNotNone(result)
        self.assertIsInstance(result, FailureType)

    def test_maybe_inject_with_zero_rate(self):
        inj = FailureInjector(seed=42, failure_rate=0.0)
        result = inj.maybe_inject(0)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
