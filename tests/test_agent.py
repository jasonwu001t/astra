"""Agent tests."""

from astra import Agent


def test_agent_init():
    agent = Agent(name="test", model="gpt-4")
    assert agent.name == "test"
    assert agent.model == "gpt-4"


def test_agent_add_tool():
    agent = Agent()
    agent.add_tool(lambda x: x)
    assert len(agent.tools) == 1

