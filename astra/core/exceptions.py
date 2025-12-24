"""Exception system"""


class AstraException(Exception):
    """Base exception class for Astra"""
    pass


class LLMException(AstraException):
    """LLM related exception"""
    pass


class AgentException(AstraException):
    """Agent related exception"""
    pass


class ConfigException(AstraException):
    """Configuration related exception"""
    pass


class ToolException(AstraException):
    """Tool related exception"""
    pass
