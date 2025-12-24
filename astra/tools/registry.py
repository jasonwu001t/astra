"""Tool registry - Native tool system"""

from typing import Optional, Any, Callable
from ..core.exceptions import AstraException
from .base import Tool


class ToolRegistry:
    """
    Tool Registry

    Provides tool registration, management and execution.
    Supports two registration methods:
    1. Tool object registration (recommended)
    2. Direct function registration (convenient)
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}

    def register_tool(self, tool: Tool):
        """
        Register Tool object

        Args:
            tool: Tool instance
        """
        if tool.name in self._tools:
            print(f"⚠️ Warning: Tool '{tool.name}' already exists, will be overwritten.")

        self._tools[tool.name] = tool
        print(f"✅ Tool '{tool.name}' registered.")

    def register_function(self, name: str, description: str, func: Callable[[str], str]):
        """
        Register function as tool (convenient method)

        Args:
            name: Tool name
            description: Tool description
            func: Tool function, accepts string param, returns string result
        """
        if name in self._functions:
            print(f"⚠️ Warning: Tool '{name}' already exists, will be overwritten.")

        self._functions[name] = {
            "description": description,
            "func": func
        }
        print(f"✅ Tool '{name}' registered.")

    def unregister(self, name: str):
        """Unregister tool"""
        if name in self._tools:
            del self._tools[name]
            print(f"🗑️ Tool '{name}' unregistered.")
        elif name in self._functions:
            del self._functions[name]
            print(f"🗑️ Tool '{name}' unregistered.")
        else:
            print(f"⚠️ Tool '{name}' does not exist.")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get Tool object"""
        return self._tools.get(name)

    def get_function(self, name: str) -> Optional[Callable]:
        """Get tool function"""
        func_info = self._functions.get(name)
        return func_info["func"] if func_info else None

    def execute_tool(self, name: str, input_text: str) -> str:
        """
        Execute tool

        Args:
            name: Tool name
            input_text: Input parameter

        Returns:
            Tool execution result
        """
        # First look for Tool object
        if name in self._tools:
            tool = self._tools[name]
            try:
                return tool.run({"input": input_text})
            except Exception as e:
                return f"Error: Exception executing tool '{name}': {str(e)}"

        # Look for function tool
        elif name in self._functions:
            func = self._functions[name]["func"]
            try:
                return func(input_text)
            except Exception as e:
                return f"Error: Exception executing tool '{name}': {str(e)}"

        else:
            return f"Error: Tool '{name}' not found."

    def get_tools_description(self) -> str:
        """
        Get formatted description string of all available tools

        Returns:
            Tool description string for building prompts
        """
        descriptions = []

        # Tool object descriptions
        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")

        # Function tool descriptions
        for name, info in self._functions.items():
            descriptions.append(f"- {name}: {info['description']}")

        return "\n".join(descriptions) if descriptions else "No tools available"

    def list_tools(self) -> list[str]:
        """List all tool names"""
        return list(self._tools.keys()) + list(self._functions.keys())

    def get_all_tools(self) -> list[Tool]:
        """Get all Tool objects"""
        return list(self._tools.values())

    def clear(self):
        """Clear all tools"""
        self._tools.clear()
        self._functions.clear()
        print("🧹 All tools cleared.")


# Global tool registry
global_registry = ToolRegistry()

