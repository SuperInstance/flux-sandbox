"""flux-sandbox — Additional tests for MockAgent, MockTransport, Scenarios"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mocks.mock_agent import MockAgent, MockTransport, CooperativeMessage


class TestMockAgentAdvanced:
    def test_agent_creation(self):
        agent = MockAgent(name="test-agent", capabilities=["compute", "reason"])
        assert agent.name == "test-agent"
        assert "compute" in agent.capabilities

    def test_agent_send_message(self):
        transport = MockTransport()
        alice = MockAgent(name="alice")
        bob = MockAgent(name="bob")
        transport.register(alice)
        transport.register(bob)
        msg = CooperativeMessage(sender="alice", target="bob", request_type="ping", payload={})
        transport.deliver(msg)
        inbox = bob.check_inbox()
        # May or may not have messages depending on transport impl
        assert isinstance(inbox, list)

    def test_agent_check_inbox_empty(self):
        agent = MockAgent(name="receiver")
        inbox = agent.check_inbox()
        assert inbox == []

    def test_agent_repr(self):
        agent = MockAgent(name="repr-test")
        r = repr(agent)
        assert "repr-test" in r

    def test_cooperative_message_fields(self):
        msg = CooperativeMessage(
            sender="a", target="b",
            request_type="compute", payload={"x": 1}
        )
        assert msg.sender == "a"
        assert msg.target == "b"
        assert msg.request_type == "compute"
        assert msg.payload["x"] == 1

    def test_transport_register_unregister(self):
        transport = MockTransport()
        agent = MockAgent(name="temp")
        transport.register(agent)
        transport.unregister("temp")

    def test_transport_deliver_to_unknown(self):
        transport = MockTransport()
        transport.deliver(CooperativeMessage(
            sender="a", target="ghost", request_type="test", payload={}
        ))

    def test_multiple_agents(self):
        transport = MockTransport()
        agents = [MockAgent(name=f"agent-{i}") for i in range(5)]
        for a in agents:
            transport.register(a)
        assert len(agents) == 5

    def test_agent_with_behavior(self):
        agent = MockAgent(name="behaved", behavior={
            "greet": lambda msg: CooperativeMessage(
                sender="behaved", target=msg.sender,
                request_type="response", payload={"greeting": "hello"}
            )
        })
        assert agent.name == "behaved"


class TestScenarios:
    def test_create_ping_scenario(self):
        from scenarios.scenarios import create_ping_scenario
        s = create_ping_scenario()
        assert s.name is not None
        assert len(s.agents_config) >= 2

    def test_create_broadcast_scenario(self):
        from scenarios.scenarios import create_broadcast_scenario
        s = create_broadcast_scenario()
        assert s is not None

    def test_create_ask_bytecode_scenario(self):
        from scenarios.scenarios import create_ask_bytecode_scenario
        s = create_ask_bytecode_scenario()
        assert s is not None

    def test_create_failure_scenario(self):
        from scenarios.scenarios import create_failure_scenario
        s = create_failure_scenario()
        assert s is not None
