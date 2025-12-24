"""Base Agent implementation."""

from typing import Any


class Agent:
    """Base agent class."""

    def __init__(self, name: str = "agent", model: str = "gpt-4"):
        self.name = name
        self.model = model
        self.tools = []
        self.memory = []

    def add_tool(self, tool: Any) -> None:
        """Register a tool."""
        self.tools.append(tool)

    def run(self, task: str) -> str:
        """Execute a task."""
        raise NotImplementedError("Subclasses must implement run()")

    def __repr__(self) -> str:
        return f"Agent(name={self.name}, model={self.model})"

