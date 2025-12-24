"""Agent tests"""

import pytest
from unittest.mock import Mock, MagicMock

from astra import SimpleAgent, Config
from astra.core.message import Message


class MockLLM:
    """Mock LLM for testing"""
    
    def __init__(self):
        self.provider = "mock"
    
    def invoke(self, messages, **kwargs):
        return "Mock response"
    
    def stream_invoke(self, messages, **kwargs):
        yield "Mock "
        yield "response"


def test_simple_agent_init():
    """Test SimpleAgent initialization"""
    llm = MockLLM()
    agent = SimpleAgent(name="test", llm=llm)
    
    assert agent.name == "test"
    assert agent.llm == llm
    assert agent.system_prompt is None
    assert len(agent._history) == 0


def test_simple_agent_with_system_prompt():
    """Test SimpleAgent with system prompt"""
    llm = MockLLM()
    agent = SimpleAgent(
        name="assistant",
        llm=llm,
        system_prompt="You are a helpful assistant."
    )
    
    assert agent.system_prompt == "You are a helpful assistant."


def test_simple_agent_run():
    """Test SimpleAgent run method"""
    llm = MockLLM()
    agent = SimpleAgent(name="test", llm=llm)
    
    response = agent.run("Hello")
    
    assert response == "Mock response"
    assert len(agent._history) == 2
    assert agent._history[0].role == "user"
    assert agent._history[0].content == "Hello"
    assert agent._history[1].role == "assistant"
    assert agent._history[1].content == "Mock response"


def test_simple_agent_history():
    """Test SimpleAgent history management"""
    llm = MockLLM()
    agent = SimpleAgent(name="test", llm=llm)
    
    agent.run("First message")
    agent.run("Second message")
    
    history = agent.get_history()
    assert len(history) == 4  # 2 user + 2 assistant
    
    agent.clear_history()
    assert len(agent._history) == 0


def test_message_creation():
    """Test Message creation"""
    msg = Message("Hello", "user")
    
    assert msg.content == "Hello"
    assert msg.role == "user"
    assert msg.timestamp is not None


def test_message_to_dict():
    """Test Message to_dict"""
    msg = Message("Hello", "assistant")
    d = msg.to_dict()
    
    assert d["role"] == "assistant"
    assert d["content"] == "Hello"


def test_config_defaults():
    """Test Config default values"""
    config = Config()
    
    assert config.default_model == "gpt-4"
    assert config.temperature == 0.7
    assert config.debug is False

