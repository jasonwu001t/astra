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

## Providers

Supports 4 LLM providers:
- **bedrock** - AWS Bedrock (default)
- **openai** - OpenAI API
- **ollama** - Local Ollama
- **vllm** - Local VLLM

## Environment Variables

```bash
# AWS Bedrock (default)
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# OpenAI
OPENAI_API_KEY=sk-xxx

# Local - Ollama
OLLAMA_HOST=http://localhost:11434/v1

# Local - VLLM
VLLM_HOST=http://localhost:8000/v1

# Generic (works with any provider)
LLM_API_KEY=your-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_ID=gpt-4

# Search tools
TAVILY_API_KEY=tvly-xxx
SERPAPI_API_KEY=xxx
```

## License

MIT
