# Smolantic AI

Smolantic AI is a Python framework leveraging `pydantic-ai` to build specialized AI agents for multi-step task processing, tool calling, and code generation/execution. It provides structured agent types with memory management and planning capabilities.

## Features

*   **Modular Agent Design:** Hierarchical agent system with `BaseAgent` as foundation, supporting specialized agents (`MultistepAgent`, `CodeAgent`).
*   **Structured Planning & Execution:** Agents follow defined steps for planning, execution, and error handling.
*   **Tool Integration:** Easily integrate and use custom or pre-built tools with specialized agents.
*   **Code Generation & Execution:** Generate and safely execute Python code using `CodeAgent` with configurable executors (local, Docker, E2B).
*   **Configuration:** Flexible configuration via environment variables or `.env` files using `pydantic-settings`.
*   **Extensible Models:** Uses Pydantic models for clear data structures (Messages, Actions, Memory).

## Agent Architecture

The framework uses a hierarchical agent system:

```
BaseAgent
├── MultistepAgent
└── CodeAgent
```

- **BaseAgent:** Provides core functionality for agent initialization, tool management, and result processing
- **MultistepAgent:** Specialized for multi-step task execution with planning capabilities
- **CodeAgent:** Specialized for code generation and execution with configurable executors

## Installation

### Production Installation

You can install Smolantic AI directly from PyPI:

```bash
pip install smolantic-ai
```

### Development Installation

For development purposes, you can install the package in editable mode:

1.  Clone the repository:
    ```bash
    git clone https://github.com/esragoth/smolantic_ai.git
    cd smolantic_ai
    ```

2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Install the package in editable mode:
    ```bash
    pip install -e .
    ```

5.  For development with testing tools:
    ```bash
    pip install -e ".[dev]"
    ```

### Environment Setup

This project requires various API keys for Large Language Models (LLMs) and external tools used by the prebuilt agents.

1.  Create a `.env` file in the root directory of the project by copying the example file:
    ```bash
    cp .env.example .env
    ```

2.  Edit the `.env` file and add your actual API keys and credentials. The required variables are listed in `.env.example` and include:
    *   **LLM Keys:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` (provide keys for the models you intend to use)
    *   **Tool Keys/URLs:**
        *   `WEATHERAPI_API_KEY` (from WeatherAPI.com)
        *   `IPGEOLOCATION_API_KEY` (from ipgeolocation.io)
        *   `EXCHANGERATE_API_KEY` (from e.g., exchangeratesapi.io)
        *   `API_NINJA_API_KEY` (from api-ninjas.com - note potential free tier limits)
        *   `JINA_API_KEY` (from Jina AI, for the reader tool)
        *   `BRIGHTDATA_PROXY_URL` (Full proxy URL including credentials, e.g., from Bright Data, for the search tool)

The application uses `pydantic-settings` to load these variables from the `.env` file. Some tool-specific keys are loaded directly using `os.getenv` within the tool functions in `src/smolantic_ai/prebuilt_tools.py`.

## Usage

Here are basic examples of how to use the agents:

**BaseAgent (Direct Usage):**

```python
import asyncio
from pydantic_ai import Tool
from smolantic_ai import BaseAgent
from smolantic_ai.models import Message, MessageRole

# Define a simple tool
def get_weather(city: str) -> str:
    """Gets the weather for a city."""
    # Replace with actual API call
    return f"The weather in {city} is sunny."

weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a specific city",
    function=get_weather,
)

async def run_base_agent():
    agent = BaseAgent(tools=[weather_tool])
    result = await agent.run("What's the weather like in London?")
    print(f"Agent Result: {result}")

asyncio.run(run_base_agent())
```

**CodeAgent:**

```python
import asyncio
from smolantic_ai import CodeAgent
from smolantic_ai.models import CodeResult

async def run_code_agent():
    # Create a CodeAgent with specific configuration
    agent = CodeAgent(
        authorized_imports=["math", "numpy"],  # Allow specific imports
        executor_type="local",  # Use local Python executor
        max_steps=10,  # Limit execution steps
        planning_interval=3  # Replan every 3 steps
    )
    
    # Run the agent with a task
    result = await agent.run(
        "Write a Python function to calculate the area of a circle given its radius."
    )
    
    # Handle the result
    if isinstance(result, CodeResult):
        print(f"Generated Code:\n{result.code}")
        print(f"Execution Output:\n{result.result}")
        if result.error:
            print(f"Error: {result.error}")
            print(f"Explanation: {result.explanation}")
            print(f"Execution Logs:\n{result.execution_logs}")

asyncio.run(run_code_agent())
```

**MultistepAgent:**

```python
import asyncio
from smolantic_ai import MultistepAgent
from smolantic_ai.models import MultistepAgentResult

async def run_multistep_agent():
    # Create a MultistepAgent with specific configuration
    agent = MultistepAgent(
        max_steps=20,  # Maximum number of steps
        planning_interval=5,  # Replan every 5 steps
        logger_name="custom_logger"  # Custom logger name
    )
    
    # Run the agent with a complex task
    result = await agent.run(
        "Research the capital of France and then find its population."
    )
    
    # Handle the result
    if isinstance(result, MultistepAgentResult):
        print(f"Final Answer: {result.result}")
        print("\nSteps Taken:")
        for i, step in enumerate(result.steps, 1):
            print(f"\nStep {i}:")
            print(f"Thought: {step.input_messages[0].content if step.input_messages else 'N/A'}")
            if step.tool_calls:
                print("Actions:")
                for tool_call in step.tool_calls:
                    print(f"  - {tool_call['name']}({tool_call['args']})")
            if step.tool_outputs:
                print("Observations:")
                for output in step.tool_outputs:
                    print(f"  - {output['name']}: {output['output']}")

asyncio.run(run_multistep_agent())
```

For more detailed examples, see:
- `examples/multistep_agent_generic.py` - Generic multistep agent usage
- `examples/multistep_agent_numbers.py` - Number processing example
- `examples/code_agent_basic.py` - Basic code generation and execution
- `examples/code_agent_advanced.py` - Advanced code agent features

## Development

To set up the development environment:

1.  Clone the repository (`git clone ...`)
2.  Create and activate a virtual environment (`python -m venv venv`, `source venv/bin/activate`)
3.  Install development dependencies:
    ```bash
    pip install -r requirements.txt
    pip install -e ".[dev]" # Installs test dependencies
    ```

## Testing

Run tests using:
```bash
pytest
```

## License

MIT License 