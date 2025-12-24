# Astra

AI Agent Framework - Forked from [Datawhale Hello-Agents](https://github.com/datawhalechina/hello-agents)

## Install

```bash
pip install -e .
```

With search tools:
```bash
pip install -e ".[search]"
```

## Quick Start

```python
from astra import AstraLLM, SimpleAgent

# Initialize LLM
llm = AstraLLM(model="gpt-4")

# Create agent
agent = SimpleAgent(name="assistant", llm=llm)

# Run
response = agent.run("Hello, who are you?")
print(response)
```

## Agent Types

- **SimpleAgent** - Basic conversational agent
- **ReActAgent** - Reasoning and acting with tool use
- **ReflectionAgent** - Self-reflection and iterative optimization
- **PlanAndSolveAgent** - Decomposition planning and step execution

## Tools

```python
from astra import ToolRegistry, CalculatorTool

registry = ToolRegistry()
registry.register_tool(CalculatorTool())

# Or register function directly
registry.register_function(
    name="greet",
    description="Greet someone",
    func=lambda name: f"Hello, {name}!"
)
```

## Structure

```
astra/
├── astra/
│   ├── core/           # Core framework (agent, llm, message, config)
│   ├── agents/         # Agent implementations
│   ├── tools/          # Tool system
│   │   └── builtin/    # Built-in tools
│   └── utils/          # Utilities
├── examples/
└── tests/
```

## Environment Variables

```bash
# OpenAI
OPENAI_API_KEY=sk-xxx

# Or generic
LLM_API_KEY=your-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_ID=gpt-4

# Local models
OLLAMA_HOST=http://localhost:11434/v1

# Search tools
TAVILY_API_KEY=tvly-xxx
SERPAPI_API_KEY=xxx
```

## License

MIT
