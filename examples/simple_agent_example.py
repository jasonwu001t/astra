"""Simple Agent Example"""

from dotenv import load_dotenv
load_dotenv()

from astra import AstraLLM, SimpleAgent


def main():
    # Initialize LLM
    llm = AstraLLM()
    
    # Create simple agent
    agent = SimpleAgent(
        name="assistant",
        llm=llm,
        system_prompt="You are a helpful AI assistant."
    )
    
    # Run conversation
    print("=== Simple Agent Demo ===\n")
    
    response = agent.run("Hello! What can you help me with?")
    print(f"Agent: {response}\n")
    
    response = agent.run("Can you explain what an AI agent is?")
    print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()

