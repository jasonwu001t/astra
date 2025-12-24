"""Reflection Agent implementation - Self-reflection and iterative optimization agent"""

from typing import Optional, List, Dict, Any
from ..core.agent import Agent
from ..core.llm import AstraLLM
from ..core.config import Config
from ..core.message import Message

# Default prompt templates
DEFAULT_PROMPTS = {
    "initial": """
Please complete the following task:

Task: {task}

Provide a complete and accurate response.
""",
    "reflect": """
Please carefully review the following response and identify potential issues or areas for improvement:

# Original Task:
{task}

# Current Response:
{content}

Analyze the quality of this response, point out shortcomings, and provide specific improvement suggestions.
If the response is already good, respond with "No improvement needed".
""",
    "refine": """
Please improve your response based on the feedback:

# Original Task:
{task}

# Previous Response:
{last_attempt}

# Feedback:
{feedback}

Provide an improved response.
"""
}


class Memory:
    """Simple short-term memory module for storing agent's action and reflection trajectory."""
    
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """Add new record to memory"""
        self.records.append({"type": record_type, "content": content})
        print(f"📝 Memory updated, added '{record_type}' record.")

    def get_trajectory(self) -> str:
        """Format all memory records into a coherent text string"""
        trajectory = ""
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"--- Previous Attempt ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- Reviewer Feedback ---\n{record['content']}\n\n"
        return trajectory.strip()

    def get_last_execution(self) -> str:
        """Get the most recent execution result"""
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return ""


class ReflectionAgent(Agent):
    """
    Reflection Agent - Self-reflection and iterative optimization agent

    Capabilities:
    1. Execute initial task
    2. Self-reflect on results
    3. Optimize based on reflection
    4. Iterate until satisfactory
    """

    def __init__(
        self,
        name: str,
        llm: AstraLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        Initialize ReflectionAgent

        Args:
            name: Agent name
            llm: LLM instance
            system_prompt: System prompt
            config: Configuration object
            max_iterations: Maximum iteration count
            custom_prompts: Custom prompt templates {"initial": "", "reflect": "", "refine": ""}
        """
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations
        self.memory = Memory()
        self.prompts = custom_prompts if custom_prompts else DEFAULT_PROMPTS
    
    def run(self, input_text: str, **kwargs) -> str:
        """
        Run Reflection Agent

        Args:
            input_text: Task description
            **kwargs: Additional parameters

        Returns:
            Final optimized result
        """
        print(f"\n🤖 {self.name} starting task: {input_text}")

        # Reset memory
        self.memory = Memory()

        # 1. Initial execution
        print("\n--- Initial attempt ---")
        initial_prompt = self.prompts["initial"].format(task=input_text)
        initial_result = self._get_llm_response(initial_prompt, **kwargs)
        self.memory.add_record("execution", initial_result)

        # 2. Iteration loop: reflect and optimize
        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1}/{self.max_iterations} ---")

            # a. Reflect
            print("\n-> Reflecting...")
            last_result = self.memory.get_last_execution()
            reflect_prompt = self.prompts["reflect"].format(
                task=input_text,
                content=last_result
            )
            feedback = self._get_llm_response(reflect_prompt, **kwargs)
            self.memory.add_record("reflection", feedback)

            # b. Check if should stop
            if "no improvement needed" in feedback.lower() or "无需改进" in feedback:
                print("\n✅ Reflection indicates no improvement needed, task complete.")
                break

            # c. Optimize
            print("\n-> Optimizing...")
            refine_prompt = self.prompts["refine"].format(
                task=input_text,
                last_attempt=last_result,
                feedback=feedback
            )
            refined_result = self._get_llm_response(refine_prompt, **kwargs)
            self.memory.add_record("execution", refined_result)

        final_result = self.memory.get_last_execution()
        print(f"\n--- Task Complete ---\nFinal result:\n{final_result}")

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_result, "assistant"))

        return final_result
    
    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """Call LLM and get complete response"""
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, **kwargs) or ""

