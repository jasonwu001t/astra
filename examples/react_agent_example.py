"""ReAct Agent Example"""

from dotenv import load_dotenv
load_dotenv()

from astra import AstraLLM, ReActAgent, ToolRegistry, CalculatorTool


def main():
    # Initialize LLM
    llm = AstraLLM()
    
    # Setup tools
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    
    # Register custom function tool
    registry.register_function(
        name="get_weather",
        description="Get weather for a city (mock data)",
        func=lambda city: f"The weather in {city} is sunny, 72°F"
    )
    
    # Create ReAct agent
    agent = ReActAgent(
        name="react_assistant",
        llm=llm,
        tool_registry=registry,
        max_steps=5
    )
    
    # Run
    print("=== ReAct Agent Demo ===\n")
    
    result = agent.run("What is 25 * 4 + 100?")
    print(f"\nFinal: {result}")


if __name__ == "__main__":
    main()

