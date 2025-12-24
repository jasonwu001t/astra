"""
Astra - AI Agent Framework

A flexible and extensible multi-agent framework built on OpenAI-compatible APIs.
"""

from .version import __version__, __author__, __email__, __description__

# Core components
from .core.llm import AstraLLM
from .core.config import Config
from .core.message import Message
from .core.exceptions import AstraException

# Agent implementations
from .agents.simple_agent import SimpleAgent
from .agents.react_agent import ReActAgent
from .agents.reflection_agent import ReflectionAgent
from .agents.plan_solve_agent import PlanAndSolveAgent

# Tool system
from .tools.registry import ToolRegistry, global_registry
from .tools.builtin.search import SearchTool, search
from .tools.builtin.calculator import CalculatorTool, calculate
from .tools.chain import ToolChain, ToolChainManager
from .tools.async_executor import AsyncToolExecutor

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__description__",

    # Core components
    "AstraLLM",
    "Config",
    "Message",
    "AstraException",

    # Agent paradigms
    "SimpleAgent",
    "ReActAgent",
    "ReflectionAgent",
    "PlanAndSolveAgent",

    # Tool system
    "ToolRegistry",
    "global_registry",
    "SearchTool",
    "search",
    "CalculatorTool",
    "calculate",
    "ToolChain",
    "ToolChainManager",
    "AsyncToolExecutor",
]
