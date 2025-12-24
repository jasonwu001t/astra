# Astra

AI Agent Framework

## Install

```bash
pip install -e .
```

## Usage

```python
from astra import Agent

agent = Agent()
response = agent.run("your task")
```

## Structure

```
astra/
├── astra/          # Core framework
│   ├── agents/     # Agent implementations
│   ├── tools/      # Tool integrations
│   ├── memory/     # Memory systems
│   └── llm/        # LLM providers
├── examples/       # Usage examples
└── tests/          # Tests
```

## License

MIT

