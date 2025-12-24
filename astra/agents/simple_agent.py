"""Simple Agent implementation"""

from typing import Optional, Iterator

from ..core.agent import Agent
from ..core.llm import AstraLLM
from ..core.config import Config
from ..core.message import Message


class SimpleAgent(Agent):
    """Simple conversational agent"""
    
    def __init__(
        self,
        name: str,
        llm: AstraLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        super().__init__(name, llm, system_prompt, config)
    
    def run(self, input_text: str, **kwargs) -> str:
        """
        Run simple agent
        
        Args:
            input_text: User input
            **kwargs: Additional parameters
            
        Returns:
            Agent response
        """
        messages = []
        
        # Add system message
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add history messages
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current user message
        messages.append({"role": "user", "content": input_text})
        
        # Call LLM
        response = self.llm.invoke(messages, **kwargs)
        
        # Save to history
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))
        
        return response
    
    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        Stream run agent
        
        Args:
            input_text: User input
            **kwargs: Additional parameters
            
        Yields:
            Agent response chunks
        """
        messages = []
        
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": input_text})
        
        # Stream LLM call
        full_response = ""
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk
        
        # Save complete conversation to history
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))

