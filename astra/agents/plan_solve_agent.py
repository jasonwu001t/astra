"""Plan and Solve Agent implementation - Decomposition planning and step-by-step execution agent"""

import ast
from typing import Optional, List, Dict
from ..core.agent import Agent
from ..core.llm import AstraLLM
from ..core.config import Config
from ..core.message import Message

# Default planner prompt template
DEFAULT_PLANNER_PROMPT = """
You are a top-tier AI planning expert. Your task is to decompose the user's complex problem into a multi-step action plan.
Ensure each step in the plan is an independent, executable sub-task, strictly arranged in logical order.
Your output must be a Python list where each element is a string describing a sub-task.

Question: {question}

Please output your plan strictly in the following format:
```python
["Step 1", "Step 2", "Step 3", ...]
```
"""

# Default executor prompt template
DEFAULT_EXECUTOR_PROMPT = """
You are a top-tier AI execution expert. Your task is to solve the problem step by step according to the given plan.
You will receive the original question, complete plan, and completed steps with results so far.
Focus on solving the "Current Step" and output only the result for that step, without extra explanations.

# Original Question:
{question}

# Complete Plan:
{plan}

# History Steps and Results:
{history}

# Current Step:
{current_step}

Please output only the answer for the "Current Step":
"""


class Planner:
    """Planner - Responsible for decomposing complex problems into simple steps"""

    def __init__(self, llm_client: AstraLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_PLANNER_PROMPT

    def plan(self, question: str, **kwargs) -> List[str]:
        """
        Generate execution plan

        Args:
            question: Problem to solve
            **kwargs: LLM call parameters

        Returns:
            List of steps
        """
        prompt = self.prompt_template.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        print("--- Generating plan ---")
        response_text = self.llm_client.invoke(messages, **kwargs) or ""
        print(f"✅ Plan generated:\n{response_text}")

        try:
            # Extract list from Python code block
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ Error parsing plan: {e}")
            print(f"Original response: {response_text}")
            return []
        except Exception as e:
            print(f"❌ Unknown error parsing plan: {e}")
            return []


class Executor:
    """Executor - Responsible for step-by-step execution"""

    def __init__(self, llm_client: AstraLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_EXECUTOR_PROMPT

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        """
        Execute task according to plan

        Args:
            question: Original question
            plan: Execution plan
            **kwargs: LLM call parameters

        Returns:
            Final answer
        """
        history = ""
        final_answer = ""

        print("\n--- Executing plan ---")
        for i, step in enumerate(plan, 1):
            print(f"\n-> Executing step {i}/{len(plan)}: {step}")
            prompt = self.prompt_template.format(
                question=question,
                plan=plan,
                history=history if history else "None",
                current_step=step
            )
            messages = [{"role": "user", "content": prompt}]

            response_text = self.llm_client.invoke(messages, **kwargs) or ""

            history += f"Step {i}: {step}\nResult: {response_text}\n\n"
            final_answer = response_text
            print(f"✅ Step {i} complete, result: {final_answer}")

        return final_answer


class PlanAndSolveAgent(Agent):
    """
    Plan and Solve Agent - Decomposition planning and step-by-step execution agent
    
    Capabilities:
    1. Decompose complex problems into simple steps
    2. Execute according to plan step by step
    3. Maintain execution history and context
    4. Arrive at final answer
    """
    
    def __init__(
        self,
        name: str,
        llm: AstraLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        Initialize PlanAndSolveAgent

        Args:
            name: Agent name
            llm: LLM instance
            system_prompt: System prompt
            config: Configuration object
            custom_prompts: Custom prompt templates {"planner": "", "executor": ""}
        """
        super().__init__(name, llm, system_prompt, config)

        if custom_prompts:
            planner_prompt = custom_prompts.get("planner")
            executor_prompt = custom_prompts.get("executor")
        else:
            planner_prompt = None
            executor_prompt = None

        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(self.llm, executor_prompt)
    
    def run(self, input_text: str, **kwargs) -> str:
        """
        Run Plan and Solve Agent
        
        Args:
            input_text: Problem to solve
            **kwargs: Additional parameters
            
        Returns:
            Final answer
        """
        print(f"\n🤖 {self.name} starting to process: {input_text}")
        
        # 1. Generate plan
        plan = self.planner.plan(input_text, **kwargs)
        if not plan:
            final_answer = "Could not generate valid action plan, task terminated."
            print(f"\n--- Task Terminated ---\n{final_answer}")
            
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))
            
            return final_answer
        
        # 2. Execute plan
        final_answer = self.executor.execute(input_text, plan, **kwargs)
        print(f"\n--- Task Complete ---\nFinal answer: {final_answer}")
        
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        
        return final_answer

